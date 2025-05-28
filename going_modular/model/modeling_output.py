import dataclasses
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union
import torch

@dataclasses.dataclass
class HeadOutputs(ModelOutput):
    r"""
    Modeling output of model in `head` package.
    
    Args:
        logits (Union[torch.Tensor, Tuple[torch.Tensor]]):
        embedding (Optional[torch.Tensor]): default is None
    """
    logits: Union[torch.Tensor, Tuple[torch.Tensor]]
    embedding: Optional[torch.Tensor] = None


@dataclasses.dataclass
class LogitsOutputs(ModelOutput):
    spectacles: torch.Tensor
    facial_hair: torch.Tensor
    pose: torch.Tensor
    emotion: torch.Tensor
    gender: torch.Tensor
    id: Union[torch.Tensor, Tuple[torch.Tensor]]

@dataclasses.dataclass
class MTLFaceForConcatEmbeddingsOutputs(ModelOutput):
    spectacles: Optional[torch.Tensor] = None
    facial_hair: Optional[torch.Tensor] = None
    pose: Optional[torch.Tensor] = None
    emotion: Optional[torch.Tensor] = None
    gender: Optional[torch.Tensor] = None
    id: Optional[torch.Tensor] = None

@dataclasses.dataclass
class MTLFaceForConcatOutputs(ModelOutput):
    r"""
    Modeling output of single task 
    """
    logits: LogitsOutputs
    embedding: Optional[MTLFaceForConcatEmbeddingsOutputs] = None

@dataclasses.dataclass
class ConcatMTLFaceRecognitionV3Outputs(ModelOutput):
    r"""
    Modeling output of concat of 3 single task
    """
    logits: LogitsOutputs
    id_embedding: Optional[torch.Tensor] = None