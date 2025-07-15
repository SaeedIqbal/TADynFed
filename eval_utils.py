import os
import json
from sklearn.calibration import calibration_curve
from client_scorer import ClientReliabilityScorer

def evaluate_calibration(scorer: ClientReliabilityScorer, output_dir: str = './results'):
    os.makedirs(output_dir, exist_ok=True)
    ece_report = {}

    for client_id in scorer.client_quality:
        preds, targets = zip(*scorer.calibration_data.get(client_id, []))
        if not preds:
            continue
        preds = torch.cat(preds, dim=0).cpu().numpy()
        targets = torch.cat(targets, dim=0).cpu().numpy()
        ece = compute_ece(preds, targets)
        ece_report[client_id] = ece

    with open(os.path.join(output_dir, 'calibration_report.json'), 'w') as f:
        json.dump(ece_report, f, indent=2)

def compute_ece(preds: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    ece = 0.0
    for i in range(preds.shape[1]):
        y_true = (targets == i)
        y_prob = preds[:, i]
        prob_true, bin_edges = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        bin_acc = np.mean(y_true)
        bin_conf = np.mean(y_prob)
        ece += abs(bin_acc - bin_conf) * (len(y_true) / preds.shape[0])
    return ece