# aggregation_module.py

import torch
import os
import json
from torch import nn

class DynamicFederatedAggregator:
    def __init__(self, global_model: nn.Module, scorer: 'ClientReliabilityScorer', param_size_bytes: int = 4):
        """
        Initialize aggregator with communication cost tracking.
        :param global_model: Global segmentation backbone
        :param scorer: Reliability scorer instance
        :param param_size_bytes: Size of one parameter in bytes (e.g., 4 for float32)
        """
        self.global_model = global_model
        self.scorer = scorer
        self.param_size_bytes = param_size_bytes
        self.communication_log = []  # Log of MB per round
        self.round_params_sent = {}  # For client-wise logging

    def aggregate(self, local_models: dict, participation_indicator: dict, round_num: int):
        """
        Aggregate client models into global using dynamic weights.
        Tracks total parameters sent per round and computes communication cost.
        """
        avg_state_dict = {}
        weights = {}

        total_params_sent = 0
        for client_id, model in local_models.items():
            if participation_indicator.get(client_id, False):
                weight = self.scorer.compute_weight(client_id, round_num)
                weights[client_id] = weight

                state = model.state_dict()
                num_params = sum(p.numel() for p in model.parameters())
                total_params_sent += num_params

                for key in self.global_model.state_dict():
                    if key not in avg_state_dict:
                        avg_state_dict[key] = weight * state[key]
                    else:
                        avg_state_dict[key] += weight * state[key]

        total_weight = sum(weights.values())
        for key in avg_state_dict:
            avg_state_dict[key] /= total_weight

        self.global_model.load_state_dict(avg_state_dict)

        # Compute and store communication cost
        comm_cost_mb = (total_params_sent * self.param_size_bytes) / (1024 ** 2)
        self.communication_log.append(comm_cost_mb)
        self.round_params_sent[round_num] = comm_cost_mb

        print(f"Round {round_num} | Total Communication Cost: {comm_cost_mb:.2f} MB")

    def save_communication_log(self, path: str):
        """Save communication cost logs to JSON file"""
        log_path = os.path.join(path, 'communication_log.json')
        with open(log_path, 'w') as f:
            json.dump({
                "communication_cost_per_round": self.round_params_sent,
                "total_communication_cost": sum(self.communication_log),
                "avg_communication_cost": sum(self.communication_log) / len(self.communication_log)
            }, f, indent=2)
        print(f"Communication cost log saved at {log_path}")