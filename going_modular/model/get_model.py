from .MTLFaceRecognition import MTLFaceRecognition, MTLFaceRecognitionForConcat
from .ConcatMTLFaceRecognition import ConcatMTLFaceRecognitionV3
import torch

from torchao.quantization.qat import (
    Int8DynActInt4WeightQATQuantizer
)
from typing import Tuple, Union

def build(
        config: dict, 
        load_checkpoint:bool = False,
        training:bool = True
    )->Tuple[ConcatMTLFaceRecognitionV3, Union[Int8DynActInt4WeightQATQuantizer, None]]:
    
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
    )

    if config['use_quant']:
        if training:
            print('add quant')
            quantizer = Int8DynActInt4WeightQATQuantizer(groupsize= 32)
            model = quantizer.prepare(model)
            return model, quantizer
        else:
            quantizer = Int8DynActInt4WeightQATQuantizer(groupsize= 32)
            model = quantizer.convert(model)
            return model, quantizer
    else:
        return model, None