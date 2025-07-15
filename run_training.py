import os
from main import train_TADynFed
from aggregation_module import DynamicFederatedAggregator
from client_scorer import ClientReliabilityScorer
from eval_utils import evaluate_calibration

TOTAL_ROUNDS = 200
OUTPUT_DIR = './results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize modules
base_model = TransBTSV2().to(device)
base_model.load_state_dict(torch.load('transbtsv2_pretrained.pth'))

client_model = TADynFedClientModel(base_model, num_tissues=3).to(device)
scorer = ClientReliabilityScorer(alpha=0.7, beta=0.3, delta=10, epsilon=0.1)
aggregator = DynamicFederatedAggregator(global_model=client_model, scorer=scorer)

optimizer = optim.AdamW(client_model.parameters(), lr=1e-4)

for round_num in range(1, TOTAL_ROUNDS + 1):
    print(f"\n[ROUND {round_num}]")
    train_TADynFed(client_loader, client_model, optimizer, scorer, aggregator, round_num)

    if round_num % 10 == 0 or round_num == TOTAL_ROUNDS:
        evaluate_calibration(scorer, output_dir=OUTPUT_DIR)
        aggregator.save_communication_log(output_dir=OUTPUT_DIR)