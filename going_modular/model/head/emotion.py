import torch
from torch import nn

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

from ..modeling_output import HeadOutputs

class EmotionDetectModule(nn.Module):
    def __init__(self):
        super(EmotionDetectModule, self).__init__()
        out_neurons = 2
        self.emotion_embedding = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.emotion_linear = nn.Linear(512, out_neurons)

    def forward(self, x_emotion)->HeadOutputs:
        x_emotion_embedding = self.emotion_embedding(x_emotion)
        x_emotion = self.emotion_linear(x_emotion_embedding)
        return HeadOutputs(
            logits= x_emotion,
            embedding= x_emotion_embedding
        )