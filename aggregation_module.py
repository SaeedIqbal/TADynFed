# aggregation_module.py

import torch
from torch import nn

class DynamicFederatedAggregator:
    def __init__(self, global_model: nn.Module, scorer: 'ClientReliabilityScorer'):
        self.global_model = global_model
        self.scorer = scorer
        self.shadow_models = {}

    def aggregate(self, local_models: Dict[str, nn.Module], participation_indicator: Dict[str, bool], round_num: int):
        """
        Aggregate client models into a global model based on dynamic weights.
        """
        avg_state_dict = {}
        weights = {}

        for client_id, model in local_models.items():
            if participation_indicator.get(client_id, False):
                score = self.scorer.compute_weight(client_id, round_num)
                weights[client_id] = score
                state = model.state_dict()
                for key in self.global_model.state_dict():
                    if key not in avg_state_dict:
                        avg_state_dict[key] = score * state[key]
                    else:
                        avg_state_dict[key] += score * state[key]

        total_weight = sum(weights.values())
        for key in avg_state_dict:
            avg_state_dict[key] /= total_weight

        self.global_model.load_state_dict(avg_state_dict)

        # Update shadow models
        for client_id in local_models:
            if client_id not in self.shadow_models or random.random() < 0.2:  # Sync every 5 rounds
                self.shadow_models[client_id] = local_models[client_id].state_dict().copy()