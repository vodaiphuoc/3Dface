import torch

from .head.id import IdRecognitionModule
from .head.gender import GenderDetectModule
from .head.emotion import EmotionDetectModule
from .head.facialhair import FacialHairDetectModule
from .head.pose import PoseDetectModule
from .head.spectacles import SpectacleDetectModule

from .backbone.mifr import create_miresnet
from .grl import GradientReverseLayer
from typing import Literal


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
from .modeling_output import (
    MTLFaceForConcatOutputs,
    LogitsOutputs,
    MTLFaceForConcatEmbeddingsOutputs,
    HeadOutputs
)

class MTLFaceRecognitionForConcat(torch.nn.Module):
    def __init__(self, backbone: BACKBONE_TYPES, num_classes:int):
        super().__init__()
        self.backbone = create_miresnet(backbone)
        
        # Head
        self.id_head = IdRecognitionModule(in_features= 512, num_classes= num_classes)
        self.gender_head = GenderDetectModule()
        self.emotion_head = EmotionDetectModule()
        self.facial_hair_head = FacialHairDetectModule()
        self.pose_head = PoseDetectModule()
        self.spectacles_head = SpectacleDetectModule()
        
    def forward(self, x, return_embedding:bool)->MTLFaceForConcatOutputs:
        (
            (x_spectacles, x_non_spectacles),
            (x_facial_hair, x_non_facial_hair),
            (x_emotion, x_non_emotion),
            (x_pose, x_non_pose),
            (x_gender, x_id)
        ) = self.backbone(x)
        
        x_spectacles: HeadOutputs = self.spectacles_head(x_spectacles, return_embedding)
        x_facial_hair: HeadOutputs = self.facial_hair_head(x_facial_hair, return_embedding)
        x_pose: HeadOutputs = self.pose_head(x_pose, return_embedding)
        x_emotion: HeadOutputs = self.emotion_head(x_emotion, return_embedding)
        x_gender: HeadOutputs = self.gender_head(x_gender, return_embedding)
        x_id: HeadOutputs = self.id_head(x_id, return_embedding)
        
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
            ) if return_embedding else None
            
        )


class MTLFaceRecognition(torch.nn.Module):


    def __init__(self, backbone:str, num_classes:int):
        super(MTLFaceRecognition, self).__init__()
        self.backbone = create_miresnet(backbone)
        
        # Head
        self.id_head = IdRecognitionModule(num_classes)
        self.gender_head = GenderDetectModule()
        self.emotion_head = EmotionDetectModule()
        self.facial_hair_head = FacialHairDetectModule()
        self.pose_head = PoseDetectModule()
        self.spectacles_head = SpectacleDetectModule()
        
        # da_discriminator (domain adaptation)
        self.da_gender_head = GenderDetectModule()
        self.da_emotion_head = EmotionDetectModule()
        self.da_facial_hair_head = FacialHairDetectModule()
        self.da_pose_head = PoseDetectModule()
        self.da_spectacles_head = SpectacleDetectModule()
        
        # grl
        self.grl_gender = GradientReverseLayer()
        self.grl_emotion = GradientReverseLayer()
        self.grl_facial_hair = GradientReverseLayer()
        self.grl_pose = GradientReverseLayer()
        self.grl_spectacles = GradientReverseLayer()
       
        
    def forward(self, x):
        (
            (x_spectacles, x_non_spectacles),
            (x_facial_hair, x_non_facial_hair),
            (x_emotion, x_non_emotion),
            (x_pose, x_non_pose),
            (x_gender, x_id)
        ) = self.backbone(x)
        
        # dt = detect
        x_spectacles = self.spectacles_head(x_spectacles)
        x_da_spectacles = self.da_spectacles_head(self.grl_spectacles(x_non_spectacles))
        
        x_facial_hair = self.facial_hair_head(x_facial_hair)
        x_da_facial_hair = self.da_facial_hair_head(self.grl_facial_hair(x_non_facial_hair))
        
        x_pose = self.pose_head(x_pose)
        x_da_pose = self.da_pose_head(self.grl_pose(x_non_pose))
        
        x_emotion = self.emotion_head(x_emotion)
        x_da_emotion = self.da_emotion_head(self.grl_emotion(x_non_emotion))
        
        x_gender = self.gender_head(x_gender)
        x_da_gender = self.da_gender_head(self.grl_gender(x_id))
        
        x_id_logits, x_id_norm = self.id_head(x_id)
        
        logits = (
                    (x_spectacles, x_da_spectacles), 
                    (x_facial_hair, x_da_facial_hair),
                    (x_pose, x_da_pose),
                    (x_emotion, x_da_emotion),
                    (x_gender, x_da_gender),
                    x_id_logits, x_id_norm
                )
        return logits
    
    
    # Trả về các task khác như bình thường trừ id chỉ trả về embedding
    def get_result(self, x):
        (
            (x_spectacles, x_non_spectacles),
            (x_facial_hair, x_non_facial_hair),
            (x_emotion, x_non_emotion),
            (x_pose, x_non_pose),
            (x_gender, x_id)
        ) = self.backbone(x)
        x_id = self.id_head.id_embedding(x_id)
        x_gender = self.gender_head(x_gender)
        x_pose = self.pose_head(x_pose)
        x_emotion = self.emotion_head(x_emotion)
        x_facial_hair = self.facial_hair_head(x_facial_hair)
        x_spectacles = self.spectacles_head(x_spectacles)
        return x_id, x_gender, x_pose, x_emotion, x_facial_hair, x_spectacles


    # Các embedding là 512 neutron
    def get_embedding(self, x):
        (
            (x_spectacles, x_non_spectacles),
            (x_facial_hair, x_non_facial_hair),
            (x_emotion, x_non_emotion),
            (x_pose, x_non_pose),
            (x_gender, x_id)
        ) = self.backbone(x)
        
        x_spectacles = self.spectacles_head.spectacle_embedding(x_spectacles)
        
        x_facial_hair = self.facial_hair_head.facial_hair_embedding(x_facial_hair)
        
        x_emotion = self.emotion_head.emotion_embedding(x_emotion)
        
        x_pose = self.pose_head.pose_embedding(x_pose)
        
        x_gender = self.gender_head.gender_embedding(x_gender)
        
        x_id = self.id_head.id_embedding(x_id)
        
        return x_spectacles, x_facial_hair, x_pose, x_emotion, x_gender, x_id
