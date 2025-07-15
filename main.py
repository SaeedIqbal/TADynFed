# main.py

import torch
import torch.optim as optim
from client_model import TADynFedClientModel
from prototype_manager import PrototypeMemoryManager
from client_scorer import ClientReliabilityScorer
from aggregation_module import DynamicFederatedAggregator
from cross_disease_adapter import DiseaseInvariantFeatureExtractor, DiseaseSpecificAdapter
from train_utils import BraTS21ClientDataset
from transbtsv2 import TransBTSV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained backbone
base_model = TransBTSV2().to(device)
base_model.load_state_dict(torch.load('transbtsv2_pretrained.pth'))

# Initialize core framework
client_model = TADynFedClientModel(base_model, num_tissues=3).to(device)

# Initialize manager and scorer
memory_manager = PrototypeMemoryManager(client_model)
scorer = ClientReliabilityScorer(alpha=0.7, beta=0.3)
aggregator = DynamicFederatedAggregator(client_model, scorer)

# Simulate client 0 with T1c and FLAIR modalities
client_dataset = BraTS21ClientDataset(client_id=0, modality_subset=['T1c', 'FLAIR'])
client_loader = DataLoader(client_dataset, batch_size=4, shuffle=True)

optimizer = optim.AdamW(client_model.parameters(), lr=1e-4)

# Cross-disease adapter
disease_invariant_extractor = DiseaseInvariantFeatureExtractor(base_model).to(device)
disease_adapters = {
    'brain_tumor': DiseaseSpecificAdapter().to(device),
    'stroke': DiseaseSpecificAdapter().to(device),
    'osteoarthritis': DiseaseSpecificAdapter().to(device)
}

# Training loop
def train_TADynFed(loader, model, disease_adapter, optimizer, memory_manager, scorer, aggregator, round_num):
    model.train()
    total_loss = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs_tensor = {m: inputs[m].to(device) for m in inputs}
        inputs_tensor['shared'] = torch.cat([inputs_tensor[m] for m in inputs_tensor], dim=1)
        labels = labels.to(device)

        seg_output, tailored_reps, shared_rep = model(inputs_tensor, list(inputs_tensor.keys()))

        # Compute losses
        loss_dict = model.get_losses(seg_output, labels, tailored_reps, shared_rep)

        # Backward pass
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        # Update client stats
        scorer.update_client_stats(client_id='client_0', quality_score=0.95, round_num=batch_idx)

        # Periodically refresh prototypes
        if batch_idx % 5 == 0:
            memory_manager.refresh_all(inputs_tensor, list(inputs_tensor.keys()))

        total_loss += loss_dict['total'].item()

    # Aggregate across clients
    local_model = model.state_dict()
    participation = {'client_0': True}
    aggregator.aggregate({'client_0': local_model}, participation, round_num=round_num)

    avg_loss = total_loss / len(loader)
    print(f"Training Loss: {avg_loss:.4f} | CLS: {loss_dict['cls'].item():.4f}, WD: {loss_dict['wd'].item():.4f}, "
          f"Cont: {loss_dict['cont'].item():.4f}, Compact: {loss_dict['compact'].item():.4f}")
    return avg_loss