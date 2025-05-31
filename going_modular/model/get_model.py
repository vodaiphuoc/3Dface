from .MTLFaceRecognition import MTLFaceRecognitionForConcat
from .ConcatMTLFaceRecognition import ConcatMTLFaceRecognitionV3
import torch
import torch.nn as nn

USAGE_LAYERS = (
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.PReLU,
    nn.Linear,
    nn.ReLU,
    nn.Sigmoid,
    nn.AdaptiveAvgPool2d,
    nn.MaxPool2d,
    nn.Flatten,
)


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

        for sub_module in model.modules():
            if isinstance(sub_module, USAGE_LAYERS):
                sub_module.weight_fake_quant = sub_module.weight_fake_quant.to(device)

        model.train()
        model = torch.ao.quantization.prepare_qat(model)
        
        return model
    else:
        model = ConcatMTLFaceRecognitionV3(
            config =config,
            load_checkpoint = load_checkpoint
        ).train()
        return model