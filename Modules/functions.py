import torch
import torch.nn.functional as F

def triplet_loss(anchor_embed, positive_embed, negative_embed, margin=1.0):
    distance_positive = F.pairwise_distance(anchor_embed, positive_embed)
    distance_negative = F.pairwise_distance(anchor_embed, negative_embed)
    loss = torch.clamp(distance_positive - distance_negative + margin, min=0.0).mean()
    return loss