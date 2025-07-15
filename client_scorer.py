# client_scorer.py

import torch
from collections import defaultdict

class ClientReliabilityScorer:
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, delta: float = 10, epsilon: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.epsilon = epsilon
        self.client_quality = defaultdict(float)
        self.client_tenure = defaultdict(int)
        self.last_update_round = defaultdict(int)

    def update_client_stats(self, client_id: str, quality_score: float, round_num: int):
        self.client_quality[client_id] = quality_score
        self.client_tenure[client_id] += 1
        self.last_update_round[client_id] = round_num

    def compute_weight(self, client_id: str, round_num: int) -> float:
        t = round_num - self.last_update_round[client_id]
        reliability = torch.exp(-t / self.delta) * (self.client_quality[client_id] + self.epsilon * self.client_tenure[client_id])
        return reliability.item()