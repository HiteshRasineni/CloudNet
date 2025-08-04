from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import config

def get_model():
    # Use the recommended way to load pretrained weights
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Optional: freeze base layers if doing feature extraction instead of fine-tuning
    # for param in model.parameters():
    #     param.requires_grad = False

    # Replace the classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),  # Helps reduce overfitting
        nn.Linear(in_features, config.NUM_CLASSES)
    )

    return model
