from tadynfed_core import (
    ModalityTailoredEncoder,
    ModalitySharedEncoder,
    TissueAwareDisentanglerHead,
    PrototypeMemoryBank
)
from losses import tissue_specific_classification_loss, wasserstein_distance_loss, contrastive_modality_loss, compactness_regularization
import torch
import torch.nn as nn

class TADynFedClientModel(nn.Module):
    MODALITIES = ['T1', 'T1c', 'T2', 'FLAIR']
    
    def __init__(self, base_model: nn.Module, num_tissues: int = 3, feature_dim: int = 256):
        super(TADynFedClientModel, self).__init__()

        # Encoders
        self.tailored_encoders = nn.ModuleDict({
            m: ModalityTailoredEncoder(base_model, m, feature_dim) for m in self.MODALITIES
        })
        self.shared_encoder = ModalitySharedEncoder(base_model, feature_dim)

        # Disentanglers
        self.disentangler_heads = nn.ModuleDict({
            m: TissueAwareDisentanglerHead(num_tissues, feature_dim) for m in self.MODALITIES
        })

        # Segmentation Head
        self.segmentation_head = nn.Sequential(
            nn.Linear(feature_dim * len(self.MODALITIES), feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_tissues),
            nn.Softmax(dim=1)
        )

        # Memory Bank per modality
        self.prototype_memory = {
            m: PrototypeMemoryBank(feature_dim=feature_dim, temperature=0.07) for m in self.MODALITIES
        }

    def forward(self, inputs: Dict[str, torch.Tensor], selected_modalities: List[str]):
        tailored_reps = {m: self.tailored_encoders[m](inputs[m]) for m in selected_modalities}
        shared_rep = self.shared_encoder(inputs['shared'])

        disentangled_reps = {
            m: self.disentangler_heads[m](tailored_reps[m]) for m in selected_modalities
        }

        fused_rep = torch.cat(list(tailored_reps.values()), dim=1)
        seg_output = self.segmentation_head(fused_rep)

        return seg_output, tailored_reps, shared_rep

    def get_losses(self, seg_logits: torch.Tensor, tissue_mask: torch.Tensor,
                   tailored_reps: Dict[str, torch.Tensor], shared_rep: torch.Tensor) -> Dict[str, torch.Tensor]:

        all_tailored = torch.stack(list(tailored_reps.values()))
        loss_cls = tissue_specific_classification_loss(seg_logits, tissue_mask)
        loss_wd = wasserstein_distance_loss(all_tailored, shared_rep)

        anchor = all_tailored[0]
        positive = all_tailored[random.choice(range(len(all_tailored)))]
        negative = all_tailored[random.choice(range(len(all_tailored)))]
        loss_cont = contrastive_modality_loss(anchor, positive, negative)

        loss_compact = torch.stack([
            compactness_regularization(
                torch.mean(rep, dim=0),
                torch.cov(rep.T),
                torch.mean(shared_rep, dim=0),
                torch.cov(shared_rep.T)
            ) for rep in tailored_reps.values()
        ]).mean()

        total_loss = (
            lambda_1 * loss_cls +
            lambda_2 * loss_wd +
            lambda_3 * loss_cont +
            lambda_4 * loss_compact
        )

        return {
            "total": total_loss,
            "cls": loss_cls,
            "wd": loss_wd,
            "cont": loss_cont,
            "compact": loss_compact
        }
#-----------------------------------------------------------------------

'''
import torch
import torch.nn as nn
from prototype_memory import PrototypeMemoryBank
from collections import defaultdict

class ModalityTailoredEncoder(nn.Module):
    def __init__(self, base_model: nn.Module, modality: str):
        super(ModalityTailoredEncoder, self).__init__()
        self.base_model = base_model
        self.modality = modality
        self.classifier = nn.Linear(256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x).mean(dim=[2,3,4])  # Global average pooling
        return self.classifier(features)


class TADynFedClientModel(nn.Module):
    MODALITIES = ['T1', 'T1c', 'T2', 'FLAIR']
    
    def __init__(self, base_model: nn.Module, num_tissues: int = 3, feature_dim: int = 256):
        super(TADynFedClientModel, self).__init__()

        # Modality-specific encoders
        self.tailored_encoders = nn.ModuleDict({
            m: ModalityTailoredEncoder(base_model, m) for m in self.MODALITIES
        })

        # Shared encoder
        self.shared_encoder = base_model

        # Prototype memory banks
        self.prototype_banks = {
            m: PrototypeMemoryBank(m, feature_dim=feature_dim) for m in self.MODALITIES
        }

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Linear(feature_dim * len(self.MODALITIES), feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_tissues),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs: dict, selected_modalities: list):
        tailored_reps = {m: self.tailored_encoders[m](inputs[m]) for m in selected_modalities}
        shared_rep = self.shared_encoder(inputs['shared']).mean(dim=[2,3,4])
        return tailored_reps, shared_rep

    def compensate_missing_modality(self, available_modalities: list) -> Dict[str, torch.Tensor]:
        """Use prototype retrieval to compensate for missing modalities"""
        compensated = {}
        all_modalities = set(self.MODALITIES)
        missing = all_modalities - set(available_modalities)

        for m in missing:
            query_emb = torch.cat([feat for feat in self.tailored_encoders.values()], dim=1)
            retrieved_feat = self.prototype_banks[m].retrieve(query_emb)
            compensated[m] = retrieved_feat

        return compensated

    def update_prototypes(self, inputs: dict, selected_modalities: list):
        """Update prototype memory bank based on current tailored encoder outputs"""
        for m in selected_modalities:
            feat = self.tailored_encoders[m](inputs[m]).detach()
            self.prototype_banks[m].update(feat.mean(dim=0))
'''