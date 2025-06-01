from .ConcatMTLFaceRecognition import ConcatMTLFaceRecognitionV3
import torch
import torch.nn as nn
from typing import Tuple, Union, Literal

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

QUANT_MODES = Literal['ptq','qat']


def build(
        config: dict, 
        load_checkpoint:bool = False,
        device: torch.device = torch.device('cuda'),
        quant_mode: QUANT_MODES = "qat"
    )->Union[ConcatMTLFaceRecognitionV3]:
    
    if config['use_quant']:
        model = ConcatMTLFaceRecognitionV3(
            config =config,
            load_checkpoint = load_checkpoint,
            backbone_quant_mode = quant_mode
        )
        if quant_mode == "qat":
            torch.backends.quantized.engine = 'qnnpack'
            model.train()
            model = torch.ao.quantization.prepare_qat(model)
            model = torch.compile(model).to(device)
        else:
            torch.backends.quantized.engine = 'qnnpack'
            model = torch.ao.quantization.prepare(model.train())
            model = model.to(device)
        
        return model
    else:
        model = ConcatMTLFaceRecognitionV3(
            config =config,
            load_checkpoint = load_checkpoint
        ).train()
        return model