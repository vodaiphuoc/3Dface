from .MTLFaceRecognition import MTLFaceRecognitionForConcat
from .ConcatMTLFaceRecognition import ConcatMTLFaceRecognitionV3
import torch

# from torchao.quantization.qat import (
#     Int8DynActInt4WeightQATQuantizer
# )
from typing import Tuple, Union

class QuantConcatMTLFaceRecognitionV3(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = ConcatMTLFaceRecognitionV3(*args, **kwargs)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self,x, *args, **kwargs):
        x = self.quant(x)
        x = self.model(x, *args, **kwargs)
        x = self.dequant(x)
        return x

def build(
        config: dict, 
        load_checkpoint:bool = False,
        training:bool = True
    )->Union[ConcatMTLFaceRecognitionV3, QuantConcatMTLFaceRecognitionV3]:
    
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
    
    if config['use_quant']:
        model = QuantConcatMTLFaceRecognitionV3(
            mtl_normalmap, 
            mtl_albedo, 
            mtl_depthmap, 
            config['num_classes']
        )
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        model = torch.ao.quantization.prepare_qat(model.train())
        return model
    else:
        model = ConcatMTLFaceRecognitionV3(
            mtl_normalmap, 
            mtl_albedo, 
            mtl_depthmap, 
            config['num_classes']
        ).train()
        return model