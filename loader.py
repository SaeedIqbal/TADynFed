import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import nibabel as nib
import random

# Root path
DATASET_ROOT = '/home/phd/datasets/'

# Supported modalities and tissue classes
MODALITIES = {
    'BraTS21': ['T1', 'T1c', 'T2', 'FLAIR'],
    'CheXpert': ['frontal', 'lateral'],  # Example modality groupings
    'Hep-2': ['DIC', 'Fluorescence']     # Two-channel microscopy
}

TISSUE_CLASSES = {
    'BraTS21': ['NCR', 'ED', 'ET'],
    'CheXpert': ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                 'Pleural Other'],
    'Hep-2': ['Homogeneous', 'Speckled', 'Nucleolar', 'Centromere',
              'Nuclear Membrane', 'Golgi']
}

# Define transformations
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ----------------------
# Dataset Classes
# ----------------------

class BraTS21Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def load_modality(self, img_path):
        # Load NIfTI MRI image (simulate multiple modalities)
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata()
        slices = [img_data[:, :, i] for i in range(img_data.shape[2])]  # Simulate 3D to 2D
        return slices

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        label_file = img_file.replace('.nii.gz', '_seg.nii.gz')

        image_path = os.path.join(self.image_dir, img_file)
        label_path = os.path.join(self.label_dir, label_file)

        image_slices = self.load_modality(image_path)
        label_slices = self.load_modality(label_path)

        slice_idx = random.randint(0, len(image_slices) - 1)
        image_slice = image_slices[slice_idx]
        label_slice = label_slices[slice_idx]

        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())

        if self.transform:
            image_slice = self.transform(image_slice)
            label_slice = self.transform(label_slice)

        return image_slice, label_slice


class CheXpertDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.image_paths = []
        self.labels = []

        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.image_paths.append(os.path.join(root, file))
                    # Simulated label from filename or annotation CSV
                    self.labels.append(random.choice([0, 1]))  # Replace with real mapping

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class Hep2Dataset(Dataset):
    def __init__(self, root_dir, split='test', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.image_paths = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Synthetic multi-channel
        label = random.choice(['Homogeneous', 'Speckled'])  # Replace with real label parsing

        if self.transform:
            image = self.transform(image)

        return image, label


class UnseenDomainDataset(Dataset):
    def __init__(self, dataset_name, root_dir, transform=None):
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.split = 'test'
        self.root_split = os.path.join(self.root_dir, self.split)
        self.file_list = []

        for root, _, files in os.walk(self.root_split):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.nii.gz')):
                    self.file_list.append(os.path.join(root, file))

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        ext = os.path.splitext(file_path)[1]

        if ext == '.nii.gz':
            img = nib.load(file_path).get_fdata()
            img = img[:, :, img.shape[-1] // 2]  # Middle slice
        else:
            img = Image.open(file_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, self.dataset_name  # No ground truth for domain adaptation evaluation


# ----------------------
# Dataloader Functions
# ----------------------

def get_braTS21_loader(batch_size=8, shuffle=True, num_workers=4):
    dataset = BraTS21Dataset(
        root_dir=os.path.join(DATASET_ROOT, 'BraTS21'),
        transform=train_transforms
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_chexpert_loader(split='valid', batch_size=16, shuffle=False, num_workers=4):
    dataset = CheXpertDataset(
        root_dir=os.path.join(DATASET_ROOT, 'CheXpert'),
        split=split,
        transform=val_transforms
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_hep2_loader(split='test', batch_size=16, shuffle=False, num_workers=4):
    dataset = Hep2Dataset(
        root_dir=os.path.join(DATASET_ROOT, 'Hep-2'),
        split=split,
        transform=val_transforms
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_unseen_domain_loader(domain_name, batch_size=16, shuffle=False, num_workers=4):
    dataset = UnseenDomainDataset(
        dataset_name=domain_name,
        root_dir=DATASET_ROOT,
        transform=val_transforms
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ----------------------
# Example Usage
# ----------------------

if __name__ == '__main__':
    braTS_loader = get_braTS21_loader(batch_size=4)
    chexpert_loader = get_chexpert_loader(split='valid')
    hep2_loader = get_hep2_loader()

    unseen_domains = ['Camelyon16', 'PANDA', 'SOKL']
    unseen_loaders = {
        domain: get_unseen_domain_loader(domain, batch_size=4)
        for domain in unseen_domains
    }

    print("BraTS21 dataloader initialized.")
    print("CheXpert & Hep-2 loaders ready for cross-domain validation.")
    print(f" {len(unseen_loaders)} unseen domain(s) loaded for generalization testing.")

    # You can now iterate through each loader
    for domain, loader in unseen_loaders.items():
        print(f"Testing on unseen domain: {domain}")
        for images, domain_name in loader:
            # Forward pass using TADynFed model
            pass