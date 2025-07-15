import torch
import torch.nn as nn

class DiseaseInvariantFeatureExtractor(nn.Module):
    def __init__(self, base_model: nn.Module):
        super(DiseaseInvariantFeatureExtractor, self).__init__()
        self.base = base_model
        self.classifier = nn.Linear(256, 256)  # Shared anatomical projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base(x)
        return self.classifier(features.mean(dim=[2,3,4]))


class DiseaseSpecificAdapter(nn.Module):
    def __init__(self, feature_dim: int = 256, num_classes: int = 3):
        super(DiseaseSpecificAdapter, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, z_inv: torch.Tensor) -> torch.Tensor:
        return self.head(z_inv)