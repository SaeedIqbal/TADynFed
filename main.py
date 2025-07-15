import torch
import torch.optim as optim
from transbtsv2 import TransBTSV2
from client_model import TADynFedClientModel
from client_scorer import ClientReliabilityScorer
from aggregation_module import DynamicFederatedAggregator
from train_utils import BraTS21ClientDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
base_model = TransBTSV2().to(device)
base_model.load_state_dict(torch.load('transbtsv2_pretrained.pth'))

# Initialize framework
client_model = TADynFedClientModel(base_model, num_tissues=3).to(device)
scorer = ClientReliabilityScorer(alpha=0.7, beta=0.3)
aggregator = DynamicFederatedAggregator(global_model=client_model, scorer=scorer)

# Simulate client 0 with T1c and FLAIR
client_dataset = BraTS21ClientDataset(client_id=0, modality_subset=['T1c', 'FLAIR'])
client_loader = DataLoader(client_dataset, batch_size=4, shuffle=True)

optimizer = optim.AdamW(client_model.parameters(), lr=1e-4)

# Training loop
def train_TADynFed(loader, model, optimizer, scorer, aggregator, round_num):
    model.train()
    total_loss = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs_tensor = {m: inputs[m].to(device) for m in inputs}
        inputs_tensor['shared'] = torch.cat([inputs_tensor[m] for m in inputs_tensor if m != 'shared'], dim=1)
        labels = labels.to(device)

        # Forward pass
        tailored_reps, shared_rep = model(inputs_tensor, list(inputs_tensor.keys()))

        # Compensate missing modalities
        available = list(inputs_tensor.keys())
        compensated = model.compensate_missing_modality(available)
        for m, rep in compensated.items():
            tailored_reps[m] = rep.to(device)

        # Fuse features
        fused_rep = torch.cat(list(tailored_reps.values()), dim=1)
        seg_output = model.segmentation_head(fused_rep)

        # Loss computation (replace with full loss module)
        loss = -torch.mean(labels * torch.log(seg_output + 1e-8))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update memory
        model.update_prototypes(inputs_tensor, list(inputs_tensor.keys()))

        # Update client stats
        scorer.update_client_stats(client_id='client_0', quality_score=0.95, round_num=batch_idx)

        # Log calibration and smoothing
        scorer.apply_temporal_smoothing(client_id='client_0', gamma=0.9)

        total_loss += loss.item()

    # Aggregate client models
    local_model = model.state_dict()
    participation = {'client_0': True}
    aggregator.aggregate({'client_0': local_model}, participation, round_num=round_num)

    avg_loss = total_loss / len(loader)
    print(f"Round {round_num} | Avg Loss: {avg_loss:.4f}")

    return avg_loss