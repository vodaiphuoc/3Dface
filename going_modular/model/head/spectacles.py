import torch
from torch import nn

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)
from ..modeling_output import HeadOutputs

class SpectacleDetectModule(nn.Module):
    def __init__(self):
        super(SpectacleDetectModule, self).__init__()
        out_neurons = 2
        self.spectacle_embedding = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.spectacle_linear = nn.Linear(512, out_neurons)

    def forward(self, x_spectacle)->HeadOutputs:
        x_spectacle_embedding = self.spectacle_embedding(x_spectacle)
        x_spectacle = self.spectacle_linear(x_spectacle_embedding)
        return HeadOutputs(
            logits= x_spectacle,
            embedding= x_spectacle_embedding
        )