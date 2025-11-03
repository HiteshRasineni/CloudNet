from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
import config

class SupConResNet50(nn.Module):
    """
    ResNet50 model with dual heads:
    - Projection head for contrastive learning (embeddings)
    - Classification head for final predictions
    """
    def __init__(self, embedding_dim=128, num_classes=None):
        super(SupConResNet50, self).__init__()
        # Load pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(in_features, embedding_dim)
        )
        
        # Classification head
        num_classes = num_classes or config.NUM_CLASSES
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input images
            return_features: If True, return both embeddings and logits
        
        Returns:
            If return_features=True: (embeddings, logits)
            Otherwise: logits (for backward compatibility)
        """
        # Extract features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Projection for contrastive learning (normalized)
        embeddings = self.projector(features)
        embeddings = F.normalize(embeddings, dim=1)
        
        # Classification logits
        logits = self.classifier(features)
        
        if return_features:
            return embeddings, logits
        return logits


def get_model(embedding_dim=128, use_supcon=True):
    """
    Get model instance
    
    Args:
        embedding_dim: Dimension of contrastive learning embeddings (default: 128)
        use_supcon: Whether to use SupCon model architecture (default: True)
    
    Returns:
        Model instance
    """
    if use_supcon:
        return SupConResNet50(embedding_dim=embedding_dim)
    else:
        # Fallback to original model for backward compatibility
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, config.NUM_CLASSES)
        )
        return model