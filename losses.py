import torch
import torch.nn.functional as F
import torch.nn as nn

def tissue_specific_classification_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -torch.mean(targets * torch.log(outputs + 1e-8))

def wasserstein_distance_loss(tailored_features: torch.Tensor, shared_features: torch.Tensor) -> torch.Tensor:
    mean_tailored = tailored_features.mean(dim=0)
    mean_shared = shared_features.mean(dim=0)
    cov_tailored = torch.cov(tailored_features.T)
    cov_shared = torch.cov(shared_features.T)
    mean_diff = torch.norm(mean_tailored - mean_shared, p=2)
    cov_diff = torch.norm(cov_tailored - cov_shared, p='fro')
    return mean_diff + cov_diff

def contrastive_modality_loss(anchor: torch.Tensor, positive: torch.Tensor,
                             negative: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    loss = torch.clamp(margin + neg_sim - pos_sim, min=0.0)
    return loss.mean()

def compactness_regularization(tailored_mean: torch.Tensor, tailored_cov: torch.Tensor,
                               shared_mean: torch.Tensor, shared_cov: torch.Tensor) -> torch.Tensor:
    mean_loss = torch.norm(tailored_mean - shared_mean, p=2)
    cov_loss = torch.norm(tailored_cov - shared_cov, p='fro')
    return mean_loss + cov_loss

def kl_divergence_loss(global_dist: torch.Tensor, local_dist: torch.Tensor) -> torch.Tensor:
    global_probs = F.softmax(global_dist, dim=1)
    local_probs = F.softmax(local_dist, dim=1)
    kl = torch.sum(global_probs * torch.log(global_probs / (local_probs + 1e-8)), dim=1).mean()
    return kl