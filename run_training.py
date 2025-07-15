# run_training.py

import os
from main import train_TADynFed
from eval_utils import evaluate_calibration, track_communication_cost

TOTAL_ROUNDS = 200
OUTPUT_DIR = './results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

for round_num in range(1, TOTAL_ROUNDS + 1):
    print(f"\n[ROUND {round_num}]")
    train_TADynFed(client_loader, client_model, optimizer, memory_manager, scorer, aggregator, round_num)

    if round_num % 10 == 0 or round_num == TOTAL_ROUNDS:
        evaluate_calibration(scorer, output_dir=OUTPUT_DIR)
        track_communication_cost(aggregator, output_dir=OUTPUT_DIR)