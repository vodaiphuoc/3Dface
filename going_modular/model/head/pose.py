import torch
from torch import nn

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)
from ..modeling_output import HeadOutputs

class PoseDetectModule(nn.Module):
    def __init__(self):
        super(PoseDetectModule, self).__init__()
        out_neurons = 2
        self.pose_embedding = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.pose_linear = nn.Linear(512, out_neurons)

    def forward(self, x_pose)->HeadOutputs:
        x_pose_embeding = self.pose_embedding(x_pose)
        x_pose = self.pose_linear(x_pose_embeding)
        return HeadOutputs(
            logits=x_pose,
            embedding= x_pose_embeding
        )