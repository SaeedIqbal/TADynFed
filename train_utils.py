# train_utils.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np

def load_nii_slice(path: str, slice_idx: Optional[int] = None) -> torch.Tensor:
    img = nib.load(path).get_fdata()
    img = (img - img.min()) / (img.max() - img.min())
    if slice_idx is None:
        slice_idx = img.shape[2] // 2
    return torch.tensor(img[:, :, slice_idx, np.newaxis], dtype=torch.float32)


class BraTS21ClientDataset(Dataset):
    def __init__(self, client_id: int, modality_subset: List[str],
                 root_dir: str = '/home/phd/datasets/BraTS21'):
        self.root_dir = root_dir
        self.modality_subset = modality_subset
        self.image_files = [f for f in os.listdir(os.path.join(root_dir, 'images')) if f.endswith('.nii.gz')]
        self.label_files = [f.replace('image', 'label') for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.root_dir, 'images', self.image_files[idx])
        label_path = os.path.join(self.root_dir, 'labels', self.label_files[idx])

        input_modalities = {m: load_nii_slice(image_path) for m in self.modality_subset}
        label = load_nii_slice(label_path)
        label = F.one_hot(label.long(), num_classes=3).float().view(-1, 3)

        return input_modalities, label