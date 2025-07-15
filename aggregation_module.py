import torch
from collections import OrderedDict
from client_scorer import ClientReliabilityScorer

class DynamicFederatedAggregator:
    def __init__(self, global_model: nn.Module, scorer: ClientReliabilityScorer):
        self.global_model = global_model.to('cpu')
        self.scorer = scorer
        self.shadow_models = {}  # Store shadow models per client
        self.round_params_sent = defaultdict(float)
        self.param_size_bytes = 4  # Assuming float32

    def aggregate(self, local_models: Dict[str, nn.Module], participation_indicator: Dict[str, bool], round_num: int):
        """
        Aggregate local model updates into a global model using adaptive weights.
        Tracks communication cost in MB.
        """
        avg_state_dict = OrderedDict()
        total_weight = 0.0

        # Compute total number of parameters
        total_params = sum(p.numel() for p in self.global_model.parameters())

        for client_id, model in local_models.items():
            if participation_indicator.get(client_id, False):
                weight = self.scorer.compute_weight(client_id, round_num)
                state = model.cpu().state_dict()

                for key in self.global_model.state_dict():
                    if key not in avg_state_dict:
                        avg_state_dict[key] = weight * state[key]
                    else:
                        avg_state_dict[key] += weight * state[key]
                total_weight += weight

                # Track communication cost
                params_sent = sum(p.numel() for p in model.parameters())
                comm_cost_mb = (params_sent * self.param_size_bytes) / (1024 ** 2)
                self.round_params_sent[round_num] += comm_cost_mb

        # Normalize by total weight
        for key in avg_state_dict:
            avg_state_dict[key] /= total_weight

        self.global_model.load_state_dict(avg_state_dict)

        # Update shadow models every 5 rounds
        if round_num % 5 == 0:
            for client_id in local_models:
                self.shadow_models[client_id] = local_models[client_id].state_dict().copy()

    def sync_shadow_model(self, client_id: str, model: nn.Module):
        """Sync shadow model if available"""
        if client_id in self.shadow_models:
            model.load_state_dict(self.shadow_models[client_id])

    def save_communication_log(self, path: str):
        """Save communication cost logs"""
        os.makedirs(path, exist_ok=True)
        log_path = os.path.join(path, "communication_cost.json")
        with open(log_path, "w") as f:
            json.dump(self.round_params_sent, f, indent=2)
        print(f"Communication cost log saved at {log_path}")