"""
Supervised Contrastive Learning Loss
Based on: "Supervised Contrastive Learning" (Khosla et al., 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    
    Args:
        temperature: Temperature scaling parameter (default: 0.07)
        base_temperature: Base temperature for scaling (default: 0.07)
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Compute supervised contrastive loss
        
        Args:
            features: Normalized feature vectors of shape [batch_size * 2, feature_dim]
            labels: Ground truth labels of shape [batch_size * 2]
        
        Returns:
            Contrastive loss scalar
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Check if we have pairs (batch_size should be even)
        # For supervised contrastive learning, we typically have 2 views per sample
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even for contrastive learning (2 views per sample)")
        
        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)  # [batch_size * 2, 1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [batch_size * 2, batch_size * 2]
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)  # [batch_size * 2, batch_size * 2]
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute exp for all pairs
        exp_logits = torch.exp(logits / self.temperature)
        
        # Mask out self-similarity
        exp_logits = exp_logits * logits_mask
        
        # Compute log_prob for positive pairs
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Sum over positive pairs for each anchor
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # Loss is negative log likelihood
        loss = -mean_log_prob_pos.mean()
        
        # Scale by base temperature
        loss = loss * (self.temperature / self.base_temperature)
        
        return loss


def create_positive_pairs(images, labels):
    """
    Create positive pairs by duplicating images and labels
    
    Args:
        images: Tensor of shape [batch_size, C, H, W]
        labels: Tensor of shape [batch_size]
    
    Returns:
        paired_images: Tensor of shape [batch_size * 2, C, H, W]
        paired_labels: Tensor of shape [batch_size * 2]
    """
    # Duplicate images and labels
    paired_images = torch.cat([images, images], dim=0)
    paired_labels = torch.cat([labels, labels], dim=0)
    
    return paired_images, paired_labels

