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

    def forward(self,x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def build(
        config: dict, 
        load_checkpoint:bool = False,
        device: torch.device = torch.device('cuda')
    )->Union[ConcatMTLFaceRecognitionV3, QuantConcatMTLFaceRecognitionV3]:
    
    if config['use_quant']:
        model = QuantConcatMTLFaceRecognitionV3(
            config =config,
            load_checkpoint = load_checkpoint
        )
        
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        model = model.to(device)

        
        _devices = set([
            _params.device 
            for _params in model.model.mtl_normalmap.backbone.parameters()
        ])
        print('check device: ', _devices)

        model.train()
        model = torch.ao.quantization.prepare_qat(model)
        model = model.to(device)
        return model
    else:
        model = ConcatMTLFaceRecognitionV3(
            config =config,
            load_checkpoint = load_checkpoint
        ).train()
        return model