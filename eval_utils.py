# eval_utils.py

import os
import json
import torch
from client_scorer import ClientReliabilityScorer
from aggregation_module import DynamicFederatedAggregator

def evaluate_calibration(scorer: ClientReliabilityScorer, output_dir: str = './results'):
    os.makedirs(output_dir, exist_ok=True)
    ece_report = {}

    for client_id in scorer.client_calibration:
        ece = scorer.compute_client_ece(client_id)
        ece_report[client_id] = ece
        print(f"Client {client_id} ECE: {ece:.4f}")
        scorer.reset_calibration(client_id)

    with open(os.path.join(output_dir, 'calibration_report.json'), 'w') as f:
        json.dump(ece_report, f, indent=2)


def track_communication_cost(aggregator: DynamicFederatedAggregator, output_dir: str = './results'):
    os.makedirs(output_dir, exist_ok=True)
    aggregator.save_communication_log(output_dir)