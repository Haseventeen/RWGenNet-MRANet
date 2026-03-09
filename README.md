

# Multi-Modal ECG Analysis and Signal Generation Framework

This repository contains the official PyTorch implementation for a dual-purpose physiological signal processing framework. It includes a **Classification Network (MRANet)** for multi-modal signal diagnosis (ECG, SCG, Resp) and a **Generative Network (Pix2Pix-based)** for 1D physiological signal translation (e.g., ECG to ABP).

## 🚀 Features

* **Multi-Modal Classification (MRANet)**
* Fuses 1D signals (ECG, SCG, Respiration) using parallel ResNet1D encoders.
* Employs a custom **TSSA Attention Module (Non-Causal)** for effective cross-modal feature interaction.
* Addresses class imbalance using a Cost-Sensitive Learning strategy with Weighted Binary Cross-Entropy (WBCE, $\mathcal{L}_{cls}$) and an automated `Target Boost Sampler`.
* Supports extensive ablation studies (e.g., `no_ecg`, `only_scg`).


* **1D Signal Generation (Pix2Pix)**
* Adapts the classic Pix2Pix architecture for 1D time-series data.
* **Generators**: Supports multiple architectures including `unet_256`, `resnet_9blocks`, and a custom `RWGenNet` (WaveNet with TSSA Attention).
* **Discriminators**: Features a sophisticated **Multi-Scale Discriminator** (`multi_scale`) utilizing Dilated Convolutions and Mish activation to capture long-range temporal dependencies.



## 📁 Repository Structure

* `MRANet.py`: Core implementation of the classification network, including dataset definitions, the ResNet1D backbone, TSSA attention, and the training loop.
* `train.py`: The main training script for the generative network (Pix2Pix).
* `pix2pix.py`: Defines the Pix2Pix model logic, loss functions (GAN, L1), and optimization steps.
* `networks.py`: Contains definitions for the Generator (`RWGenNet`, `UnetGenerator`, etc.) and basic blocks.
* `discriminator.py`: Implements the 1D PatchGAN and the Multi-Scale Discriminator.
* `base_model.py`: Abstract base class handling model saving, loading, and learning rate scheduling.

## 🛠️ Installation

Requirements:

* Python >= 3.8
* PyTorch >= 1.10
* NumPy
* SciPy
* scikit-learn
* matplotlib
* tqdm

```bash
pip install torch numpy scipy scikit-learn matplotlib tqdm

```

## 📊 Data Preparation

The models expect pre-processed 1D physiological signals saved as `.pt` (PyTorch Tensor) files containing dictionaries.

### For Classification (MRANet)

The dataset should be a `.pt` file containing a dictionary with the following keys:

* `'ECG'`: Tensor of shape `(N, Length)`
* `'SCG'`: Tensor of shape `(N, Length)`
* `'Generated_RESP'`: Tensor of shape `(N, Length)`
* `'label'`: Tensor of shape `(N, 6)` for one-hot encoded labels.

### For Generation (Pix2Pix)

The dataset should contain:

* `'ecg'`: Source signal tensor (e.g., ECG).
* `'breath'`: Target signal tensor (e.g., ABP or Respiration).

## 🏃‍♂️ Training and Evaluation

### 1. Classification Network (MRANet)

To train the multi-modal classification network, run `MRANet.py`.

```bash
python MRANet.py

```

**Key Configurations in `MRANet.py`:**

* `MODE_NAME`: Set to `''` for full modalities, or choose ablation modes (`'no_ecg'`, `'only_scg'`, etc.).
* `pos_weights`: Adjust the tensor in `ECGClassifier.__init__` to balance your specific dataset distribution.
* Ensure `train_path` and `test_path` in `prepare_data()` point to your `.pt` files.

Outputs:

* `best_model.pth`
* `ecg_classifier_final.pth`
* `training_history.png`

### 2. Generative Network (Pix2Pix)

To train the signal translation network, execute `train.py`.

```bash
python train.py

```

**Key Configurations in `train.py` (via `TestOptions`):**

* `netG`: Choose the generator architecture (`'RWGenNet'`, `'unet_256'`, etc.).
* `netD`: Choose the discriminator architecture (`'multi_scale'`, `'dilated'`, etc.).
* `dataset_mode`: Set data paths (`./data/train.pt`, etc.).
* `batch_size`, `n_epochs`, `lr`.

Outputs will be saved in the directory specified by `opt.checkpoints_dir` (default: `./checkpoints`).

## 🧠 Architecture Details

### TSSA Attention Module

Both networks heavily rely on the **TSSA (Temporal Spatial Self-Attention)** module. In MRANet, it operates across the modality dimension (Batch, Modality=3, Features=512) to dynamically weight the importance of ECG, SCG, and Resp signals based on their feature representations.

### Multi-Scale Discriminator

In the generative task, the discriminator (`MultiScaleDiscriminator`) employs an ensemble of three `NLayerDiscriminator1D` instances with varying base channel capacities (`ndf`, `ndf*2`, `ndf*4`). This allows the model to scrutinize the generated 1D signals at different representation levels simultaneously.