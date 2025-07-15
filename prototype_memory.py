import torch
import torch.nn.functional as F
from typing import Dict, List, Union

class PrototypeMemoryBank:
    def __init__(self, modality: str, max_size: int = 200, feature_dim: int = 256, temperature: float = 0.07):
        self.modality = modality
        self.max_size = max_size
        self.temperature = temperature
        self.prototypes = torch.randn(max_size, feature_dim)

    def update(self, new_prototype: torch.Tensor):
        """Update FIFO queue with new prototype"""
        new_prototype = new_prototype.detach().cpu()
        if len(self.prototypes) >= self.max_size:
            self.prototypes = torch.cat([new_prototype.unsqueeze(0), self.prototypes[:-1]])
        else:
            self.prototypes = torch.cat([new_prototype.unsqueeze(0), self.prototypes])

    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """Retrieve most similar prototypes using cosine similarity"""
        sim = F.cosine_similarity(query_embedding.unsqueeze(1), self.prototypes.unsqueeze(0), dim=-1)
        _, indices = sim.topk(top_k, dim=1)
        return self.prototypes[indices].mean(dim=1)

    def bayesian_retrieval(self, query_embedding: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """Bayesian retrieval strategy that incorporates uncertainty in distance-based matching"""
        distances = torch.norm(query_embedding.unsqueeze(1) - self.prototypes.unsqueeze(0), dim=-1)
        weights = F.softmax(-distances / sigma, dim=1)
        weighted_prototypes = torch.matmul(weights, self.prototypes)
        return weighted_prototypes.mean(dim=0)