# prototype_manager.py

from client_model import TADynFedClientModel
from tadynfed_core import PrototypeMemoryBank
import torch

class PrototypeMemoryManager:
    def __init__(self, model: TADynFedClientModel):
        self.model = model
        self.memory_banks = model.prototype_memory  # Dictionary of banks per modality
        self.refresh_interval = 5  # Refresh every 5 rounds

    def refresh_all(self, inputs: dict, selected_modalities: list):
        """Refresh memory bank using current tailored encoder outputs"""
        with torch.no_grad():
            for m in selected_modalities:
                if m in self.model.tailored_encoders:
                    feat = self.model.tailored_encoders[m](inputs[m])
                    self.memory_banks[m].update(feat.mean(dim=0), client_id=hash(m))

    def compensate_missing_modality(self, query_modality: str, available_features: torch.Tensor) -> torch.Tensor:
        """Retrieve most similar prototype to compensate missing modality"""
        return self.memory_banks[query_modality].retrieve(available_features)