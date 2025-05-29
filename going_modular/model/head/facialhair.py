import torch
from torch import nn

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

from ..modeling_output import HeadOutputs

class FacialHairDetectModule(nn.Module):
    def __init__(self):
        super(FacialHairDetectModule, self).__init__()
        out_neurons = 2
        self.facial_hair_embedding = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.facial_hair_linear = nn.Linear(512, out_neurons)

    def forward(self, x_facial_hair)->HeadOutputs:
        x_facial_hair_embedding = self.facial_hair_embedding(x_facial_hair)
        x_facial_hair = self.facial_hair_linear(x_facial_hair_embedding)
        return HeadOutputs(
            logits= x_facial_hair,
            embedding= x_facial_hair_embedding
        )