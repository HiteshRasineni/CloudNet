# model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
import config

class SupConResNet(nn.Module):
    """ResNet50 backbone + projection head for SupCon"""
    def __init__(self, feat_dim=128):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)

        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.out_dim = base_model.fc.in_features

        # Projection head (as in SupCon paper)
        self.projection_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, feat_dim),
        )

        # Classification head (used only during fine-tuning)
        self.classifier = nn.Linear(self.out_dim, config.NUM_CLASSES)

    def forward(self, x, proj=True):
        feat = self.encoder(x).squeeze()

        if proj:  # contrastive learning forward
            return nn.functional.normalize(self.projection_head(feat), dim=1)
        else:     # classification forward
            return self.classifier(feat)


def get_model():
    return SupConResNet()
