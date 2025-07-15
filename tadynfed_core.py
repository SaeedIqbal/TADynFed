import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

class ModalityTailoredEncoder(nn.Module):
    def __init__(self, base_model: nn.Module, modality_name: str, feature_dim: int = 256):
        super(ModalityTailoredEncoder, self).__init__()
        self.base_model = base_model
        self.modality = modality_name
        self.feature_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        feat_flat = features.mean(dim=[2,3,4])  # Global average pooling
        return self.feature_proj(feat_flat)


class ModalitySharedEncoder(nn.Module):
    def __init__(self, base_model: nn.Module, feature_dim: int = 256):
        super(ModalitySharedEncoder, self).__init__()
        self.base_model = base_model
        self.shared_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        shared_feat = features.mean(dim=[2,3,4])
        return self.shared_proj(shared_feat)


class TissueAwareDisentanglerHead(nn.Module):
    def __init__(self, num_tissues: int, feature_dim: int = 256):
        super(TissueAwareDisentanglerHead, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(num_tissues, feature_dim))

    def forward(self, tailored_feat: torch.Tensor) -> torch.Tensor:
        relevance_scores = torch.matmul(self.attention_weights, tailored_feat.t())
        relevance_scores = torch.softmax(relevance_scores / (feature_dim ** 0.5), dim=0)
        disentangled_feats = relevance_scores.unsqueeze(2) * tailored_feat.unsqueeze(0)
        return disentangled_feats


class PrototypeMemoryBank(nn.Module):
    def __init__(self, max_size: int = 200, feature_dim: int = 256, temperature: float = 0.07):
        super(PrototypeMemoryBank, self).__init__()
        self.max_size = max_size
        self.temperature = temperature
        self.register_buffer('prototypes', torch.randn(max_size, feature_dim))
        self.register_buffer('client_weights', torch.ones(max_size))  # For prototype aging

    def update(self, new_prototype: torch.Tensor, client_id: int):
        idx = client_id % self.max_size
        self.prototypes[idx] = new_prototype.detach()
        self.client_weights[idx] = client_id  # Track last updated by

    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        sim = torch.cosine_similarity(query.unsqueeze(1), self.prototypes.unsqueeze(0), dim=-1)
        _, indices = sim.topk(top_k, dim=1)
        return self.prototypes[indices].mean(dim=1)