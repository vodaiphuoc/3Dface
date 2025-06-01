import torch

from .attr_head import AttributesDetectModule, IdRecognitionModule
from .backbone.mifr import MIResNet, QuantMIResNet
from .backbone.mifr import create_miresnet
from .grl import GradientReverseLayer
from typing import Literal, Union


# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

BACKBONE_TYPES = Literal[
    'miresnet18', 
    'miresnet34', 
    'miresnet50', 
    'miresnet101', 
    'miresnet152'
]

MAPTYPE_KEYS = Literal[
    "normalmap",
    "albedo",
    "depthmap"
]

BACKBONE_FREEZE = Literal[
    'no',
    "all",
    "layer1",
    "layer2",
    "layer3",
]

from .modeling_output import (
    MTLFaceForConcatOutputs,
    LogitsOutputs,
    MTLFaceForConcatEmbeddingsOutputs,
    HeadOutputs
)

from huggingface_hub import hf_hub_download

class MTLFaceRecognitionForConcat(torch.nn.Module):
    def __init__(
            self, 
            config: dict,
            load_checkpoint:bool,
            mapkey: MAPTYPE_KEYS,
            backbone_quant_mode: Literal['ptq','qat','no'] = 'no'
        ):
        super().__init__()

        backbone: BACKBONE_TYPES = config['backbone']
        num_classes = config['num_classes']
        freeze_options: BACKBONE_FREEZE = config['freeze_options']

        self.backbone: Union[MIResNet, QuantMIResNet] = create_miresnet(backbone, backbone_quant_mode)
        self.backbone_quant_mode = backbone_quant_mode
        # Head
        self.id_head = IdRecognitionModule(in_features= 512, num_classes= num_classes)
        self.gender_head = AttributesDetectModule()
        self.emotion_head = AttributesDetectModule()
        self.facial_hair_head = AttributesDetectModule()
        self.pose_head = AttributesDetectModule()
        self.spectacles_head = AttributesDetectModule()

        if load_checkpoint:
            self._load_backbone_ckpt(mapkey)

        self._freeze_layers(freeze_options)

    def _load_backbone_ckpt(self, mapkey: MAPTYPE_KEYS):
        cache_ckpt_path = hf_hub_download(
            "Daiphuoc/3DFaceCheckpoints", 
            repo_type="model",
            filename = "checkpoint.pth",
            subfolder= f"{mapkey}/models",
        )
        ckpt = torch.load(cache_ckpt_path, map_location= 'cpu')
        
        init_backbone_state_dict = self.backbone.state_dict()
        
        backbone_state_dict = {}
        for k,v in ckpt['model_state_dict'].items():
            current_key = k
            if "backbone" in k:
                current_key = current_key.replace("backbone.",'')

            if self.backbone_quant_mode != "no":
                current_key = "model." + current_key
            
            
            if init_backbone_state_dict.get(current_key) is not None:
                backbone_state_dict[current_key] = v

        print('post process key',backbone_state_dict.keys())

        try:
            self.backbone.load_state_dict(backbone_state_dict)
            print(f'success in loading ckpt {mapkey} for backbone')
        except Exception as e:
            print(f'error in loading ckpt {mapkey} for backbone')
            try:
                self.backbone.load_state_dict(backbone_state_dict,strict= False)
                print(f'success in re-load ckpt {mapkey} for backbone with strict is False')
            except Exception as e:
                print(f'error in re-load ckpt {mapkey} for backbone with strict is False')
    
    def _freeze_layers(self, freeze_options: BACKBONE_FREEZE):
        r"""
        Freeze layers in backbone
        """
        if freeze_options == "all":
            for prams in self.backbone.parameters():
                prams.requires_grad = False
        elif freeze_options == 'no':
            for prams in self.backbone.parameters():
                prams.requires_grad = True
        else:
            freeze_layer_names = []
            for name, sub_module in self.backbone.named_modules():
                if freeze_options[:-1] + str(int(freeze_options[-1]) + 1) in name:
                    break
                else:
                    freeze_layer_names.append(name)

            for name, sub_module in self.backbone.named_modules():
                if name in freeze_layer_names:
                    for prams in sub_module.parameters():
                        prams.requires_grad = False
                else:
                    for prams in sub_module.parameters():
                        prams.requires_grad = True

    def forward(self, x:torch.Tensor)->MTLFaceForConcatOutputs:
        (
            (x_spectacles, x_non_spectacles),
            (x_facial_hair, x_non_facial_hair),
            (x_emotion, x_non_emotion),
            (x_pose, x_non_pose),
            (x_gender, x_id)
        ) = self.backbone(x)

        x_spectacles: HeadOutputs = self.spectacles_head(x_spectacles)
        x_facial_hair: HeadOutputs = self.facial_hair_head(x_facial_hair)
        x_pose: HeadOutputs = self.pose_head(x_pose)
        x_emotion: HeadOutputs = self.emotion_head(x_emotion)
        x_gender: HeadOutputs = self.gender_head(x_gender)
        x_id: HeadOutputs = self.id_head(x_id)
        
        return MTLFaceForConcatOutputs(
            logits= LogitsOutputs(
                x_spectacles.logits,
                x_facial_hair.logits,
                x_pose.logits,
                x_emotion.logits,
                x_gender.logits,
                x_id.logits
            ),
            embedding = MTLFaceForConcatEmbeddingsOutputs(
                x_spectacles.embedding,
                x_facial_hair.embedding,
                x_pose.embedding,
                x_emotion.embedding,
                x_gender.embedding,
                x_id.embedding
            )
            
        )


# class MTLFaceRecognition(torch.nn.Module):


#     def __init__(self, backbone:str, num_classes:int):
#         super(MTLFaceRecognition, self).__init__()
#         self.backbone = create_miresnet(backbone)
        
#         # Head
#         self.id_head = IdRecognitionModule(num_classes)
#         self.gender_head = GenderDetectModule()
#         self.emotion_head = EmotionDetectModule()
#         self.facial_hair_head = FacialHairDetectModule()
#         self.pose_head = PoseDetectModule()
#         self.spectacles_head = SpectacleDetectModule()
        
#         # da_discriminator (domain adaptation)
#         self.da_gender_head = GenderDetectModule()
#         self.da_emotion_head = EmotionDetectModule()
#         self.da_facial_hair_head = FacialHairDetectModule()
#         self.da_pose_head = PoseDetectModule()
#         self.da_spectacles_head = SpectacleDetectModule()
        
#         # grl
#         self.grl_gender = GradientReverseLayer()
#         self.grl_emotion = GradientReverseLayer()
#         self.grl_facial_hair = GradientReverseLayer()
#         self.grl_pose = GradientReverseLayer()
#         self.grl_spectacles = GradientReverseLayer()
       
        
#     def forward(self, x):
#         (
#             (x_spectacles, x_non_spectacles),
#             (x_facial_hair, x_non_facial_hair),
#             (x_emotion, x_non_emotion),
#             (x_pose, x_non_pose),
#             (x_gender, x_id)
#         ) = self.backbone(x)
        
#         # dt = detect
#         x_spectacles = self.spectacles_head(x_spectacles)
#         x_da_spectacles = self.da_spectacles_head(self.grl_spectacles(x_non_spectacles))
        
#         x_facial_hair = self.facial_hair_head(x_facial_hair)
#         x_da_facial_hair = self.da_facial_hair_head(self.grl_facial_hair(x_non_facial_hair))
        
#         x_pose = self.pose_head(x_pose)
#         x_da_pose = self.da_pose_head(self.grl_pose(x_non_pose))
        
#         x_emotion = self.emotion_head(x_emotion)
#         x_da_emotion = self.da_emotion_head(self.grl_emotion(x_non_emotion))
        
#         x_gender = self.gender_head(x_gender)
#         x_da_gender = self.da_gender_head(self.grl_gender(x_id))
        
#         x_id_logits, x_id_norm = self.id_head(x_id)
        
#         logits = (
#                     (x_spectacles, x_da_spectacles), 
#                     (x_facial_hair, x_da_facial_hair),
#                     (x_pose, x_da_pose),
#                     (x_emotion, x_da_emotion),
#                     (x_gender, x_da_gender),
#                     x_id_logits, x_id_norm
#                 )
#         return logits
    
    
#     # Trả về các task khác như bình thường trừ id chỉ trả về embedding
#     def get_result(self, x):
#         (
#             (x_spectacles, x_non_spectacles),
#             (x_facial_hair, x_non_facial_hair),
#             (x_emotion, x_non_emotion),
#             (x_pose, x_non_pose),
#             (x_gender, x_id)
#         ) = self.backbone(x)
#         x_id = self.id_head.id_embedding(x_id)
#         x_gender = self.gender_head(x_gender)
#         x_pose = self.pose_head(x_pose)
#         x_emotion = self.emotion_head(x_emotion)
#         x_facial_hair = self.facial_hair_head(x_facial_hair)
#         x_spectacles = self.spectacles_head(x_spectacles)
#         return x_id, x_gender, x_pose, x_emotion, x_facial_hair, x_spectacles


#     # Các embedding là 512 neutron
#     def get_embedding(self, x):
#         (
#             (x_spectacles, x_non_spectacles),
#             (x_facial_hair, x_non_facial_hair),
#             (x_emotion, x_non_emotion),
#             (x_pose, x_non_pose),
#             (x_gender, x_id)
#         ) = self.backbone(x)
        
#         x_spectacles = self.spectacles_head.spectacle_embedding(x_spectacles)
        
#         x_facial_hair = self.facial_hair_head.facial_hair_embedding(x_facial_hair)
        
#         x_emotion = self.emotion_head.emotion_embedding(x_emotion)
        
#         x_pose = self.pose_head.pose_embedding(x_pose)
        
#         x_gender = self.gender_head.gender_embedding(x_gender)
        
#         x_id = self.id_head.id_embedding(x_id)
        
#         return x_spectacles, x_facial_hair, x_pose, x_emotion, x_gender, x_id
