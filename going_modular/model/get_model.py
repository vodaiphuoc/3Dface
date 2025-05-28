from .MTLFaceRecognition import MTLFaceRecognition, MTLFaceRecognitionForConcat
from .ConcatMTLFaceRecognition import ConcatMTLFaceRecognitionV3
import torch
from torchao.quantization import (
    quantize_,
    Int8DynamicActivationInt4WeightConfig
)
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    IntXQuantizationAwareTrainingConfig,
)

from huggingface_hub import hf_hub_download


def build(
        config: dict, 
        load_checkpoint:bool = False,
        training:bool = True
    ):
    
    mtl_normalmap = MTLFaceRecognitionForConcat(config['backbone'], config['num_classes'])
    if load_checkpoint:
        cache_ckpt_path1 = hf_hub_download(
            "Daiphuoc/3DFaceCheckpoints", 
            repo_type="model",
            filename = "checkpoint.pth",
            subfolder= "normalmap/models",
        )
        checkpoint_1 = torch.load(cache_ckpt_path1, map_location= 'cpu')
        mtl_normalmap.load_state_dict(checkpoint_1['model_state_dict'])

    
    mtl_albedo = MTLFaceRecognitionForConcat(config['backbone'], config['num_classes'])
    if load_checkpoint:
        cache_ckpt_path2 = hf_hub_download(
            "Daiphuoc/3DFaceCheckpoints", 
            repo_type="model",
            filename = "checkpoint.pth",
            subfolder= "albedo/models",
        )
        checkpoint_2 = torch.load(cache_ckpt_path2, map_location= 'cpu')
        try:
            mtl_albedo.load_state_dict(checkpoint_2['model_state_dict'])
        except Exception as e:
            print('error loading state dict albedo')
    
    mtl_depthmap = MTLFaceRecognitionForConcat(config['backbone'], config['num_classes'])
    if load_checkpoint:
        cache_ckpt_path3 = hf_hub_download(
            "Daiphuoc/3DFaceCheckpoints", 
            repo_type="model",
            filename = "checkpoint.pth",
            subfolder= "depthmap/models",
        )
        checkpoint_3 = torch.load(cache_ckpt_path3, map_location= 'cpu')
        mtl_depthmap.load_state_dict(checkpoint_3['model_state_dict'])

    model = ConcatMTLFaceRecognitionV3(mtl_normalmap, mtl_albedo, mtl_depthmap, config['num_classes']).to(torch.float32)

    if config['use_quant']:
        if training:
            activation_config = FakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
            weight_config = FakeQuantizeConfig(torch.int4, group_size=32)
            quantize_(
                model,
                IntXQuantizationAwareTrainingConfig(activation_config,weight_config)
            )
        else:
            quantize_(model, Int8DynamicActivationInt4WeightConfig(group_size= 32))
    
    return model