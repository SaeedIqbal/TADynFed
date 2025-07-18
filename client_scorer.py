'''
import torch
from collections import defaultdict
from sklearn.calibration import calibration_curve
import numpy as np

class ClientReliabilityScorer:
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, delta: float = 10, epsilon: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.epsilon = epsilon
        self.client_quality = defaultdict(float)          # Data quality score
        self.client_tenure = defaultdict(int)            # Participation duration
        self.last_update_round = defaultdict(int)         # Last active round
        self.client_calibration = defaultdict(list)       # Store predicted probabilities and true labels
        self.client_ece = defaultdict(float)             # Track ECE per client

    def update_client_stats(self, client_id: str, quality_score: float, round_num: int):
        self.client_quality[client_id] = quality_score
        self.client_tenure[client_id] += 1
        self.last_update_round[client_id] = round_num

    def log_calibration(self, client_id: str, probs: torch.Tensor, labels: torch.Tensor):
        """Log predictions and labels for ECE computation"""
        probs_np = probs.detach().cpu().numpy()
        labels_np = labels.argmax(dim=1).cpu().numpy() if len(labels.shape) > 1 else labels.cpu().numpy()

        # Store for batch-wise ECE calculation
        self.client_calibration[client_id].extend([
            (p, l) for p, l in zip(probs_np, labels_np)
        ])

    def compute_weight(self, client_id: str, round_num: int) -> float:
        t = round_num - self.last_update_round[client_id]
        reliability = np.exp(-t / self.delta) * (self.client_quality[client_id] + self.epsilon * self.client_tenure[client_id])
        return float(reliability)

    def compute_client_ece(self, client_id: str, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE) for a client"""
        if not self.client_calibration[client_id]:
            return float('nan')

        preds, targets = zip(*self.client_calibration[client_id])
        preds = np.vstack(preds)
        targets = np.array(targets)

        ece = 0.0
        for i in range(preds.shape[1]):
            y_true = (targets == i)
            y_prob = preds[:, i]
            _, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
            bin_acc = np.mean(y_true)
            bin_conf = np.mean(y_prob)
            bin_count = len(y_true)
            ece += abs(bin_acc - bin_conf) * (bin_count / len(targets))

        self.client_ece[client_id] = ece
        return ece

    def reset_calibration(self, client_id: str):
        self.client_calibration[client_id].clear()

#--------------------------------------------------------------
'''
#=====================================================================================================
import torch
import os
from typing import Dict, List, Optional
from collections import defaultdict

class ClientReliabilityScorer:
    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 delta: float = 10, epsilon: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.epsilon = epsilon
        self.client_quality = defaultdict(float)          # Q_k(t): data quality score
        self.client_tenure = defaultdict(int)            # T_k(t): duration of active participation
        self.last_active_round = defaultdict(int)        # t_k^last: last round client was active
        self.smoothed_quality = defaultdict(float)       # \tilde{Q}_k(t): smoothed quality
        self.reliability_scores = defaultdict(float)
        self.dropout_prob = 0.2                        # p_drop: random exclusion probability

    def update_client_stats(self, client_id: str, quality_score: float, round_num: int):
        self.client_quality[client_id] = quality_score
        self.client_tenure[client_id] += 1
        self.last_active_round[client_id] = round_num

    def compute_weight(self, client_id: str, round_num: int) -> float:
        t_gap = round_num - self.last_active_round[client_id]
        decay_factor = torch.exp(-t_gap / self.delta).item()
        reliability = decay_factor * (self.client_quality[client_id] + self.epsilon * self.client_tenure[client_id])
        self.reliability_scores[client_id] = reliability
        return reliability

    def apply_temporal_smoothing(self, client_id: str, gamma: float = 0.9):
        self.smoothed_quality[client_id] = gamma * self.client_quality[client_id] + \
                                           (1 - gamma) * self.smoothed_quality.get(client_id, self.client_quality[client_id])

    def is_client_dropout(self, client_id: str) -> bool:
        return random.random() < self.dropout_prob * (1 - self.reliability_scores[client_id])