# main.py

import torch
import torch.optim as optim
from client_model import TADynFedClientModel
from prototype_manager import PrototypeMemoryManager
from client_scorer import ClientReliabilityScorer, ClientCalibrationMonitor
from aggregation_module import DynamicFederatedAggregator
from train_utils import BraTS21ClientDataset
from cross_disease_adapter import DiseaseInvariantFeatureExtractor, DiseaseSpecificAdapter
from transbtsv2 import TransBTSV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
base_model = TransBTSV2().to(device)
base_model.load_state_dict(torch.load('transbtsv2_pretrained.pth'))

# Initialize core framework
client_model = TADynFedClientModel(base_model, num_tissues=3).to(device)

# Initialize managers
memory_manager = PrototypeMemoryManager(client_model)
scorer = ClientReliabilityScorer(alpha=0.7, beta=0.3)
aggregator = DynamicFederatedAggregator(client_model, scorer)

# Simulate client with T1c and FLAIR
client_dataset = BraTS21ClientDataset(client_id=0, modality_subset=['T1c', 'FLAIR'])
client_loader = torch.utils.data.DataLoader(client_dataset, batch_size=4, shuffle=True)

optimizer = optim.AdamW(client_model.parameters(), lr=1e-4)

# Training loop
def train_TADynFed(loader, model, optimizer, memory_manager, scorer, aggregator, round_num):
    model.train()
    total_loss = 0

    for inputs, labels in loader:
        inputs_tensor = {m: inputs[m].to(device) for m in inputs}
        inputs_tensor['shared'] = torch.cat([inputs_tensor[m] for m in inputs_tensor], dim=1)
        labels = labels.to(device)

        seg_output, tailored_reps, shared_rep = model(inputs_tensor, list(inputs_tensor.keys()))

        # Losses
        loss_dict = model.get_losses(seg_output, labels, tailored_reps, shared_rep)

        # Backward pass
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        # Update client stats
        scorer.update_client_stats(client_id='client_0', quality_score=0.95, round_num=round_num)
        scorer.log_calibration(client_id='client_0', probs=seg_output, labels=labels)

        # Periodically refresh prototypes
        if round_num % 5 == 0:
            memory_manager.refresh_all(inputs_tensor, list(inputs_tensor.keys()))

        total_loss += loss_dict['total'].item()

    # Aggregate client models
    local_model = model.state_dict()
    participation = {'client_0': True}
    aggregator.aggregate({'client_0': local_model}, participation, round_num=round_num)

    avg_loss = total_loss / len(loader)
    print(f"Training Loss: {avg_loss:.4f} | CLS: {loss_dict['cls'].item():.4f}, WD: {loss_dict['wd'].item():.4f}, "
          f"Cont: {loss_dict['cont'].item():.4f}, Compact: {loss_dict['compact'].item():.4f}")

    return avg_loss