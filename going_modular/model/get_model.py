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

def build(
        config: dict, 
        load_checkpoint:bool = False,
        training:bool = True
    ):
    
    mtl_normalmap = MTLFaceRecognitionForConcat(
        backbone= config['backbone'], 
        num_classes= config['num_classes'],
        load_checkpoint= load_checkpoint,
        mapkey="normalmap",
        freeze_options=config['freeze_options']
    )
    
    mtl_albedo = MTLFaceRecognitionForConcat(
        backbone= config['backbone'], 
        num_classes= config['num_classes'],
        load_checkpoint= load_checkpoint,
        mapkey= "albedo",
        freeze_options=config['freeze_options']
    )
    
    mtl_depthmap = MTLFaceRecognitionForConcat(
        backbone= config['backbone'], 
        num_classes= config['num_classes'],
        load_checkpoint= load_checkpoint,
        mapkey= "depthmap",
        freeze_options=config['freeze_options']
    )
    
    model = ConcatMTLFaceRecognitionV3(
        mtl_normalmap, 
        mtl_albedo, 
        mtl_depthmap, 
        config['num_classes']
    ).to(torch.float32)

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