
## 🔬 TADynFed: Dynamic Modality-Adaptive Federated Learning with Tissue-Aware Disentanglement

This repository contains the official implementation of:

> **"Dynamic Modality-Adaptive Federated Learning with Tissue-Aware Disentanglement for Cross-Disease Segmentation"**

The proposed method introduces a novel framework that addresses the challenges of **modality heterogeneity**, **non-IID data distribution**, and **cross-disease generalization** in decentralized medical imaging environments. This work builds upon the methodologies from:
- **HistoFL**: A federated learning approach for histopathology image classification.
- **PointTransformerFL (PntTranForFL)**: A modality-decoupled FL framework for multi-modal MRI segmentation.

TADynFed enhances these frameworks by introducing:
- **Tissue-aware disentanglement**
- **Prototype memory-based modality compensation**
- **Client reliability scoring**
- **Cross-disease knowledge transfer**

---

## 🧪 Overview

| Feature | Description |
|--------|-------------|
| **Framework** | PyTorch |
| **Training Datasets** | BraTS21, CheXpert, Hep-2 |
| **Validation Datasets** | Camelyon16, PANDA, SOKL |
| **Key Innovations** | Tissue-aware disentanglement, prototype distillation, adaptive client aggregation, cross-disease transfer |
| **Model Architecture** | Transformer-based encoder-decoder (TransBTSV2 backbone) |

---

## 📁 Dataset Details

### 1. **BraTS21 (Brain Tumor Segmentation)**
- **Modality**: T1, T1c, T2, FLAIR
- **Tissue Classes**: Enhancing Tumor (ET), Peritumoral Edema (ED), Necrotic Core (NCR)
- **Data Size**: ~2040 patient cases
- **Description**: Large-scale brain tumor dataset with pixel-level annotations and diverse scanner types.
- **Reference**:  
  Menze BH et al., *BraTS 2021: Multimodal Brain Tumor Image Segmentation Benchmark*, arXiv preprint.

### 2. **CheXpert (Thoracic Disease Diagnosis)**
- **Modality**: Chest X-ray (frontal/lateral views)
- **Class Labels**:  
  "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other"
- **Data Size**: 224,000 images from 10,002 patients
- **Reference**:  
  Irvin J et al., *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels for Common Thoracic Diseases*, NeurIPS 2019.

### 3. **Hep-2 (HEp-2 Cell Pattern Classification)**
- **Modality**: Fluorescence microscopy (DIC, IF)
- **Class Labels**: Homogeneous, Speckled, Nucleolar, Centromere, Nuclear Membrane, Golgi
- **Data Size**: Thousands of HEp-2 cell images
- **Description**: Microscopy dataset used to classify autoimmune disease-related staining patterns.
- **Reference**:  
   Foggia P, Percannella G, Soda P, et al. Benchmarking HEp-2 cells classification methods[J]. Medical Imaging, IEEE Transactions on, 2013 [https://qixianbiao.github.io/HEp2Cell/]

---

## 🔍 Unseen Domain Validation Datasets

### 1. **Camelyon16 (Histopathological Cancer Detection)**
- **Modality**: Whole-slide histopathology images
- **Task**: Breast cancer metastasis detection
- **Reference**:  
  Ehteshami Bejnordi B et al., *Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer*, JAMA, 2017.

### 2. **PANDA (Prostate Cancer Grading)**
- **Modality**: Histopathological whole-slide images
- **Task**: Gleason score prediction
- **Reference**:  
  Litjens G et al., *A Survey on Deep Learning in Medical Image Analysis*, Medical Image Analysis, 2017.

### 3. **SOKL (Synthetic Osteoarthritis Knee Cartilage Dataset)**
- **Modality**: Synthetic MRI-like imaging
- **Task**: Cartilage structure segmentation
---

## 🧠 Framework Components

TADynFed integrates four core components to address key challenges in mix-modal federated learning (MixMFL):

### 1. **Tissue-Aware Modality Disentanglement**
Each modality is encoded using both a **modality-tailored** and a **modality-shared** encoder. Class-specific attention masks are applied during training to enforce disentanglement per tissue class.

```python
z_{i,l}^{\text{disentangled}, m} = \text{Softmax}\left(\frac{M_l \cdot z_i^{\text{tailored}, m}}{\sqrt{F}}\right)
```

- **Functionality**: Improves fusion robustness when clients have missing modalities.
- **Code File**: `client_model.py`, `tadynfed_core.py`

---

### 2. **Dynamic Prototype Memory with Global Consistency**
Maintains a FIFO bank of learned prototypes per modality. Clients lacking certain modalities can retrieve similar features from other banks using cosine similarity or Bayesian retrieval.

```python
z_m = \arg\max_{q_c \in M_m} \text{sim}(f_{\text{query}}, q_c)
```
```python
z_m^{\text{Bayes}} = \mathbb{E}_{z \sim p(z|M_m)}\left[ \text{softmax}\left(-\frac{\|f_{\text{query}} - z\|^2}{\sigma^2}\right) \cdot z \right]
```

- **Functionality**: Compensates for missing inputs and ensures consistency across clients.
- **Code File**: `prototype_memory.py`

---

### 3. **Adaptive Aggregation for Dynamic Clients**
Clients contribute dynamically based on their data quality $ Q_k(t) $ and tenure $ T_k(t) $. We use a weighted aggregation rule:

```python
w_k(t) = \frac{\alpha Q_k + \beta \log(1+\tau_k(t))}{\sum (\cdot)}
```

Additionally, we apply temporal smoothing and dropout handling:

```python
\tilde{Q}_k(t) = \gamma Q_k(t) + (1 - \gamma)\tilde{Q}_k(t - 1)
```

- **Functionality**: Ensures fair and stable convergence under dynamic participation.
- **Code File**: `client_scorer.py`

---

### 4. **Cross-Disease Knowledge Transfer**
Encodes global anatomical patterns via a **disease-invariant feature extractor**, and adapts them using **disease-specific heads**:

```python
z = z_{\text{inv}} + z_d
```

Where:
- $ z_{\text{inv}} $: Shared representation
- $ z_d $: Pathology-specific embedding

- **Functionality**: Enables accurate segmentation even on unseen diseases.
- **Code File**: `cross_disease_adapter.py`

---

## 📦 Code Structure

```
TADynFed/
├── tadynfed_core.py             # Core framework modules
├── client_model.py              # Client model definition
├── prototype_memory.py          # Prototype management
├── client_scorer.py             # Client reliability and calibration scoring
├── aggregation_module.py        # Weighted aggregation and shadow models
├── train_utils.py               # Data loading and preprocessing
├── losses.py                    # Multi-objective loss functions
├── cross_disease_adapter.py     # Disease-invariant head adapter
├── eval_utils.py                # Evaluation and metric tracking
├── main.py                      # Main training loop
└── run_training.py              # Training runner script
```

---

## 🧩 Key Functionalities

### ✅ **Modality-Tailored & Shared Encoders**
- Learns modality-specific and shared representations
- Used in `client_model.py` and `tadynfed_core.py`

### ✅ **Tissue-Aware Disentangler Head**
- Applies softmax-based attention masks per tissue class
- Found in `tadynfed_core.py`

### ✅ **Prototype Memory Bank**
- Stores and retrieves modality-specific prototypes
- Implements similarity-based and Bayesian retrieval strategies
- Located in `prototype_memory.py`

### ✅ **Client Reliability Scoring**
- Computes weights using data quality and participation duration
- Temporal smoothing and dropout handling
- Implemented in `client_scorer.py`

### ✅ **Federated Aggregation Module**
- Adaptive averaging using client weights
- Shadow model synchronization every 5 rounds
- Communication cost tracking
- Defined in `aggregation_module.py`

### ✅ **Cross-Disease Generalization**
- Disease-invariant feature extraction
- Disease-specific head adaptation
- Found in `cross_disease_adapter.py`

### ✅ **Multi-Objective Loss Functions**
Includes:
- Tissue-specific classification loss
- Wasserstein distance-based disentanglement
- Contrastive modality loss
- Compactness regularization
- KL-divergence-based prototype distillation

```python
\mathcal{L}_{\text{TADynFed}} = \lambda_1 \mathcal{L}_{\text{cls}} + \lambda_2 \mathcal{L}_{\text{wd}} + \lambda_3 \mathcal{L}_{\text{cont}} + \lambda_4 \mathcal{L}_{\text{compact}}
```

- Found in `losses.py`

---

## 🚀 How to Use

### 🧾 Requirements

```bash
pip install torch torchvision numpy scikit-learn nibabel matplotlib seaborn
```

### 📂 Dataset Setup

Ensure your dataset root is structured as follows:

```
/home/phd/datasets/
├── BraTS21/
│   ├── images/
│   └── labels/
├── CheXpert/
│   ├── train/
│   └── valid/
├── Hep-2/
│   ├── images/
│   └── masks/
├── Camelyon16/
│   └── test/
├── PANDA/
│   └── test/
└── SOKL/
    └── test/
```

### ⚙️ Train the Model

```bash
cd TADynFed
python run_training.py
```

You can configure:
- Number of clients
- Modalities per client
- Hyperparameters (`alpha`, `beta`, `gamma`, `delta`, `epsilon`)
- Backbone architecture (`TransBTSV2`, `nnUNet`, etc.)

---

## 📈 Results Summary

| Method | mDice ↑ | ASD ↓ | HD95 ↓ | ECE ↓ | CommCost ↓ |
|--------|---------|-------|----------|--------|-------------|
| FedAvg | 53.06% | 2.43 mm | 11.38 mm | 0.14 | 100 MB |
| FedProx | 54.48% | 2.35 mm | 11.02 mm | 0.14 | 105 MB |
| IOP-FL | 55.10% | 2.30 mm | 10.65 mm | 0.13 | 102 MB |
| FedAAAI | 54.03% | 2.35 mm | 10.81 mm | 0.13 | 101 MB |
| HistoFL | 59.28% | 2.07 mm | 10.14 mm | 0.13 | 95 MB |
| PntTranForFL | 63.30% | 2.03 mm | 9.97 mm | 0.11 | 95 MB |
| **TADynFed (Ours)** | **66.03%** | **1.85 mm** | **8.70 mm** | **0.09** | **76 MB** |

These results show significant improvements over existing methods in:
- Segmentation accuracy (mDice)
- Boundary alignment (ASD, HD95)
- Calibration stability (ECE)
- Communication efficiency (CommCost)

---

## 📌 Citation

If you find this code useful in your research, please cite our manuscript:

```bibtex
@article{iqbaltadynfed,
  title={Dynamic Modality-Adaptive Federated Learning with Tissue-Aware Disentanglement for Cross-Disease Segmentation},
  author={Iqbal, Saeed and Zhang, Xiaopin and Khan, Muhammad Attique and Liu, Weixiang and Almujally, Nouf Abdullah and Hussain, Amir},
  journal={Preprint submitted to Elsevier},
  year={2025}
}
```

---

## 🧪 Future Work

We plan to extend this framework by:
- Introducing **Bayesian uncertainty estimation**
- Supporting **fully 3D volumetric segmentation**
- Integrating **federated class-incremental learning**
- Validating on **real-world clinical trials**

---

## 🤝 Contact

For questions, collaborations, or requests, please contact:

> Saeed Iqbal  
> saeediqbalkhattak@gmail.com  

---

## 📚 References

1. Menze BH et al. (2021). *BraTS 2021: Multimodal Brain Tumor Image Segmentation Benchmark*. arXiv preprint.
2. Irvin J et al. (2019). *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels for Common Thoracic Diseases*. NeurIPS.
3. Quan TM et al. (2018). *HEp-2 Cell Classification Using Deep Learning on Indirect Immunofluorescence Images*. IEEE TMI.
4. Ehteshami Bejnordi B et al. (2017). *Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer*. JAMA.
5. Litjens G et al. (2017). *A Survey on Deep Learning in Medical Image Analysis*. Medical Image Analysis.
6. Zhang Y et al. (2022). *Synthetic Medical Image Generation via Deep Learning*. Nature Biomedical Engineering.
7. Huang W et al. (2022). *EEFed: Personalized Federated Learning of Execution & Evaluation Dual Network for CPS Intrusion Detection*. IEEE TIFS.
8. Jiang M et al. (2023). *IOP-FL: Inside-Outside Personalization for Federated Medical Image Segmentation*. IEEE TMI.
9. Li D et al. (2024). *FedDiff: Diffusion Model Driven Federated Learning for Multi-Modal and Multi-Clients*. IEEE TCSVT.
10. Yu S et al. (2024). *Robust Multimodal Federated Learning for Incomplete Modalities*. Computer Communications.

---

## 📎 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🧱 Acknowledgments

This work was supported through **Princess Nourah bint Abdulrahman University Researchers Supporting Project number (PNURSP2025R410)**.

---

## 🧩 Contributing

Contributions, bug reports, and feature requests are welcome! Please submit issues or pull requests for enhancements.

---

Let me know if you'd like this README exported as a `.md` file, or extended with:
- Installation instructions
- Pretrained model download links
- Custom configuration guide
- Reproduction commands for ablation studies and figures

I'm happy to tailor it further for public release or peer review!
