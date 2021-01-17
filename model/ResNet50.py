import torch.nn as nn
from torchvision import models


class ResNet50Features(nn.Module):
    def __init__(self):
        super(ResNet50Features, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

    def forward(self, x):
        return self.resnet(x)
