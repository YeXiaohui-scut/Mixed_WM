# 🔐 Latent-WOFA-Seal

**Robust Watermarking System Combining MetaSeal, WOFA, and Latent-Space Embedding**

A state-of-the-art anti-attack watermarking system that operates in the Stable Diffusion latent space, combining semantic watermark generation (MetaSeal), progressive distortion training (WOFA), and single-channel latent injection (LW/MarkPlugger).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Citation](#citation)

---

## 🎯 Overview

Latent-WOFA-Seal is a PyTorch-based watermarking framework that embeds robust, semantic watermarks into images at the latent space level. The system is designed to resist various attacks including:

- ✂️ **Cropping/Partial Theft** - Through self-attention holographic properties
- 🔄 **Geometric Transformations** - Rotation, scaling, translation
- 🎨 **Image Quality Degradation** - Noise, compression
- 🖼️ **Semantic Edits** - Content-aware modifications

### Core Technologies

1. **MetaSeal** - Semantic watermark generation using BLIP captions + QR codes
2. **WOFA** - Progressive distortion curriculum learning strategy
3. **LW/MarkPlugger** - Single-channel latent space watermark injection

---

## ✨ Key Features

- 🧠 **Self-Attention Encoder** - Holographic properties for crop resistance
- 📈 **Progressive Training** - WOFA curriculum learning (0.0 → 1.0 strength)
- 🎯 **Zero-Initialized Embedder** - Preserves image quality during training start
- 🔍 **U-Net Extractor** - Separates weak watermark from strong semantic background
- 📊 **TensorBoard Integration** - Real-time training visualization
- ⚡ **Mixed Precision Training** - Automatic Mixed Precision (AMP) support
- 🔧 **Modular Design** - Easy to extend and customize

---

## 🏗️ Architecture

### Two-Stage Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage I: Robustness Pre-training (WOFA)                        │
│  ┌──────┐    ┌────┐    ┌──────────┐    ┌────┐    ┌──────┐     │
│  │  QR  │ -> │ Ep │ -> │ Distort  │ -> │ Dp │ -> │ QR'  │     │
│  └──────┘    └────┘    └──────────┘    └────┘    └──────┘     │
│  256×256     64×64      Progressive     64×64     256×256      │
│                         Strength↑                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Stage II: Quality & Fusion Training                            │
│  ┌─────┐    ┌─────┐   ┌────┐   ┌──────────┐   ┌────┐          │
│  │Image│ -> │ VAE │-->│  C │-->│ Distort  │-->│ DC │          │
│  └─────┘    └─────┘   └────┘   └──────────┘   └────┘          │
│              Frozen     Embed   Fixed/Random   Extract         │
│                                                                  │
│  ┌──────┐   ┌────┐                                              │
│  │  QR  │-->│ Ep │ (Frozen)                                    │
│  └──────┘   └────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Model Components

| Component | Input | Output | Function |
|-----------|-------|--------|----------|
| **Pattern Encoder (Ep)** | QR [1×256×256] | Latent [1×64×64] | Encode with Self-Attention |
| **Pattern Decoder (Dp)** | Latent [1×64×64] | QR [1×256×256] | U-Net reconstruction |
| **Latent Embedder (C)** | Image [4×64×64] + Pattern [1×64×64] | Watermarked [4×64×64] | Zero-init fusion |
| **Latent Extractor (DC)** | Watermarked [4×64×64] | Pattern [1×64×64] | Small U-Net extraction |

---

## 🔧 Installation

### Requirements

- Python >= 3.8
- CUDA >= 11.7 (for GPU acceleration)
- PyTorch >= 2.0

### Setup

```bash
# Clone repository
git clone https://github.com/YeXiaohui-scut/Mixed_WM.git
cd Mixed_WM

# Create virtual environment
conda create -n wofa python=3.9
conda activate wofa

# Install dependencies
pip install -r requirements.txt

# Download COCO dataset (example)
mkdir -p data/coco
cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip annotations_trainval2017.zip
```

---

## 🚀 Quick Start

### Configuration

Edit `config.py` to set your paths and hyperparameters:

```python
# Data paths
COCO_ROOT = "./data/coco/train2017"
COCO_ANN_FILE = "./data/coco/annotations/captions_train2017.json"

# Training settings
STAGE1_EPOCHS = 100
STAGE1_BATCH_SIZE = 16
STAGE2_EPOCHS = 50
STAGE2_BATCH_SIZE = 8

# Enable/disable BLIP
USE_BLIP = True  # Set to False for faster testing
```

### Stage I Training

```bash
python train_stage1.py
```

This will:
- Train Pattern Encoder and Decoder with WOFA progressive distortion
- Save checkpoints to `checkpoints/stage1_YYYYMMDD_HHMMSS/`
- Log to TensorBoard in `logs/stage1_YYYYMMDD_HHMMSS/`

### Monitor Training

```bash
tensorboard --logdir=logs/
```

Navigate to `http://localhost:6006` to view:
- Loss curves
- Distortion strength progression
- Visualization: Original QR → Latent → Distorted → Reconstructed

---

## 📊 Training

### Stage I: Robustness Pre-training

**Goal**: Train robust QR encoder/decoder resistant to latent-space attacks

**Training Loop**:
```python
for epoch in range(EPOCHS):
    strength = epoch / EPOCHS  # Progressive: 0.0 → 1.0
    
    latent_pattern = Encoder(QR)
    distorted_pattern = Distort(latent_pattern, strength)
    reconstructed_QR = Decoder(distorted_pattern)
    
    loss = MSE(QR, reconstructed_QR)
```

**Key Mechanisms**:
- **Progressive Distortion**: Masking ratio, rotation angle, noise std all scale with `strength`
- **Self-Attention**: Middle ResBlocks use multi-head attention for holographic properties
- **Curriculum Learning**: Easy (strength=0) → Hard (strength=1)

### Stage II: Quality & Fusion Training

**Goal**: Embed watermarks into images while preserving quality

**Training Loop**:
```python
# Frozen components: Ep, Dp, VAE
with torch.no_grad():
    image_latent = VAE.encode(image)
    pattern_latent = Encoder(QR)

watermarked_latent = Embedder(image_latent, pattern_latent)
distorted_latent = Distort(watermarked_latent, random_strength)
extracted_pattern = Extractor(distorted_latent)

# Decoupled losses
loss_quality = MSE(image_latent, watermarked_latent)
loss_recovery = MSE(pattern_latent, extracted_pattern)
loss = loss_quality + 10.0 * loss_recovery
```

**Key Features**:
- **Zero Initialization**: Embedder output layer initialized to 0
- **Decoupled Loss**: Separate quality and recovery objectives
- **Fixed/Random Distortion**: Can use fixed or random strength

---

## 📁 Project Structure

```
Mixed_WM/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.py                      # Configuration & hyperparameters
├── data_provider.py              # COCO dataset + MetaSeal watermark generation
├── distortions.py                # WOFA progressive distortion layer
├── train_stage1.py               # Stage I training script
├── train_stage2.py               # Stage II training script (to be implemented)
├── inference.py                  # Watermark embedding/extraction inference
│
├── models/
│   ├── __init__.py
│   ├── pattern_encoder.py        # Ep: QR → Latent (with Self-Attention)
│   ├── pattern_decoder.py        # Dp: Latent → QR (U-Net)
│   ├── latent_embedder.py        # C: Embed pattern into image latent
│   ├── latent_extractor.py       # DC: Extract pattern from latent
│   └── full_model.py             # Complete model integration
│
├── checkpoints/                   # Model checkpoints (timestamped)
│   ├── stage1_YYYYMMDD_HHMMSS/
│   └── stage2_YYYYMMDD_HHMMSS/
│
└── logs/                          # TensorBoard logs (timestamped)
    ├── stage1_YYYYMMDD_HHMMSS/
    └── stage2_YYYYMMDD_HHMMSS/
```

---

## 🔬 Technical Details

### Self-Attention Mechanism

```python
class SelfAttention(nn.Module):
    """
    Provides holographic properties: local features contain global information.
    Critical for crop/masking resistance.
    """
    def __init__(self, channels, num_heads=8):
        self.attention = nn.MultiheadAttention(channels, num_heads)
```

**Why it matters**: When 50% of latent is masked, remaining features still encode full QR information.

### Progressive Distortion

```python
# Masking area ratio
mask_ratio = 0.1 + strength * 0.6  # 10% → 70%

# Rotation angle
rotation = strength * 45.0  # 0° → 45°

# Noise std
noise_std = strength * 0.1  # 0.0 → 0.1
```

### Zero Initialization

```python
class ZeroInitializedConv(nn.Conv2d):
    def __init__(self, ...):
        super().__init__(...)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
```

**Effect**: `watermarked_latent ≈ image_latent` at training start → stable quality.

---

## 📈 Monitoring & Visualization

### TensorBoard Logs

Stage I visualization includes:
1. **Original QR** - Input binary QR codes
2. **Latent Pattern** - Encoded representation (64×64)
3. **Distorted Pattern** - After progressive distortion
4. **Reconstructed QR** - Decoder output
5. **Difference Map** - Reconstruction error visualization

### Checkpoint Management

Checkpoints are saved with timestamps:
```
checkpoints/
└── stage1_20251119_094615/
    ├── checkpoint_epoch_0005.pth
    ├── checkpoint_epoch_0010.pth
    ├── best_model.pth
    └── latest.pth
```

---

## 🎓 Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@article{metaseal2024,
  title={MetaSeal: Semantic Watermarking with BLIP and QR Codes},
  author={...},
  journal={...},
  year={2024}
}

@article{wofa2024,
  title={WOFA: Progressive Distortion for Robust Watermarking},
  author={...},
  journal={...},
  year={2024}
}

@article{latent_watermark2024,
  title={Latent-Space Watermarking for Diffusion Models},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgments

- **BLIP**: Salesforce/blip-image-captioning-base
- **Stable Diffusion**: stabilityai/sd-vae-ft-mse
- **COCO Dataset**: Microsoft COCO Consortium

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: eeyxh2023@mail.scut.edu.cn

---

**Built with ❤️ by the Latent-WOFA-Seal Team**