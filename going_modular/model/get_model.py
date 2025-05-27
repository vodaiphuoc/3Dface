from .MTLFaceRecognition import MTLFaceRecognition
from .ConcatMTLFaceRecognition import ConcatMTLFaceRecognitionV3
import torch
from torchao.quantization import (
    quantize_,
    Int8DynamicActivationInt4WeightConfig,
)
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    FromIntXQuantizationAwareTrainingConfig,
    IntXQuantizationAwareTrainingConfig,
)


def build(
        config: dict, 
        use_quant: bool = False,
        load_checkpoint:bool = False
    ):
    
    mtl_normalmap = MTLFaceRecognition(config['backbone'], config['num_classes'])
    if load_checkpoint:
        checkpoint_1 = torch.load(config['checkpoint_1'])
        mtl_normalmap.load_state_dict(checkpoint_1['model_state_dict'])

    
    mtl_albedo = MTLFaceRecognition(config['backbone'], config['num_classes'])
    if load_checkpoint:
        checkpoint_2 = torch.load(config['checkpoint_2'])
        mtl_albedo.load_state_dict(checkpoint_2['model_state_dict'])

    
    mtl_depthmap = MTLFaceRecognition(config['backbone'], config['num_classes'])
    
    if load_checkpoint:
        checkpoint_3 = torch.load(config['checkpoint_3'])
        mtl_depthmap.load_state_dict(checkpoint_3['model_state_dict'])

    model = ConcatMTLFaceRecognitionV3(mtl_normalmap, mtl_albedo, mtl_depthmap, config['num_classes'])

    if use_quant:
        activation_config = FakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
        weight_config = FakeQuantizeConfig(torch.int8, group_size=32)
        quantize_(
            model,
            IntXQuantizationAwareTrainingConfig(activation_config, weight_config),
        )
    return model