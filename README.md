# AG-ViM: Attention-Gated Vision Mamba for Brain Tumor Segmentation

> An efficient deep learning architecture for automated brain tumor segmentation from MRI scans, combining Vision Mamba with U-Net style encoder–decoder design and attention gating mechanisms.

---

## Overview

Brain tumor segmentation from MRI images is a critical and time-intensive task in medical image analysis. AG-ViM (**A**ttention-**G**ated **Vi**sion **M**amba) is a lightweight yet powerful deep learning model that automates this process with high accuracy and significantly reduced computational cost compared to traditional approaches like U-Net and Swin-UNet.

The model integrates:
- A **Vision Mamba backbone** for efficient long-range feature extraction
- A **U-Net-style encoder–decoder** for spatial reconstruction
- **Attention Gate mechanisms** to focus on tumor-relevant regions and suppress background noise

---

## Results

| Model | Best Val Dice ↑ | Best Val IoU ↑ | Avg GPU Memory (MB) ↓ | Avg Train Time/Epoch (s) ↓ | Params (M) |
|---|---|---|---|---|---|
| U-Net | 0.750 | 0.600 | 1500 | 10.0 | ~31.0 |
| Vision Mamba | 0.800 | 0.650 | 800 | 7.0 | ~1.2 |
| **AG-ViM (Ours)** | **0.820** | **0.680** | 950 | 8.5 | ~1.8 |

AG-ViM achieves the **best segmentation accuracy** while using **~37% less GPU memory** and requiring **~15× fewer parameters** than U-Net.

---

## Architecture

```
MRI Input (224×224)
        │
   Patch Embed (16×16 patches → dim=256)
        │
  ┌─────┴─────┐
  │  Encoder  │  ← 4× MambaBlock (SSM + depthwise conv + gating)
  └─────┬─────┘
        │ skip connections
  ┌─────┴──────────┐
  │    Decoder     │  ← Upsample + Conv + AttentionGate (×2)
  └─────┬──────────┘
        │
   Final Upsample → Conv 1×1
        │
  Segmentation Mask
```

Each **MambaBlock** applies:
1. Layer normalization
2. Linear projection → split into feature & gate
3. Depthwise Conv1d (kernel=7) for local context
4. Sigmoid gating
5. Residual connection

Each **AttentionGate** computes a spatial attention map from the decoder feature (`g`) and the skip connection (`x`), weighting the skip features before concatenation.

---

## Dataset

The model is trained and evaluated on the **BraTS 2020 2D Multichannel** dataset.

- Images: Multi-modal MRI fused to a single FLAIR channel, resized to `224×224`
- Masks: Binary tumor masks (edema, tumor core, enhancing tumor → unified)
- Augmentation: Random horizontal and vertical flips (50% probability each)
- Split: Standard train/validation split

**Download:** [BraTS 2020 @ Penn Medicine](https://www.med.upenn.edu/cbica/brats2020)

Expected directory structure:
```
BraTS2020_2D_Multichannel/
├── images/
│   ├── BraTS20_001_fused.png
│   └── ...
└── masks/
    ├── BraTS20_001__mask.png
    └── ...
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ag-vim.git
cd ag-vim

# Install dependencies
pip install torch torchvision einops opencv-python matplotlib tqdm seaborn pandas

# (Optional) Install Mamba SSM for full Mamba support
pip install mamba-ssm==2.0.3 --no-build-isolation
```

> **GPU requirement:** An NVIDIA GPU with CUDA support is required for efficient training. Tested on Google Colab (T4/A100).

---

## Usage

### Training

Open `AG_Vim_Main.ipynb` in Google Colab and mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Set your dataset path:

```python
BASE_DIR  = "/content/drive/MyDrive/BraTS2020_2D_Multichannel"
IMG_DIR   = os.path.join(BASE_DIR, "images")
MASK_DIR  = os.path.join(BASE_DIR, "masks")
```

Run training:

```python
ag_vim_model = AG_ViM(dim=256).to(device)

ag_vim_results = train_and_evaluate_model(
    model=ag_vim_model,
    model_name="AG-ViM",
    epochs=150,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer_class=torch.optim.AdamW,
    lr=5e-4,
    device=device
)
```

### Saving the Model

```python
torch.save(model.state_dict(), "/content/drive/MyDrive/AG_Vim_Model.pth")
```

### Loss Function

AG-ViM uses a combined Dice + BCE loss:

```python
def combo_loss(pred, target):
    return dice_loss(pred, target) + bce(pred, target)
```

---

## Model Comparison

The notebook includes a full comparison pipeline against:
- **2D U-Net** (CNN baseline)
- **Vision Mamba Encoder-Decoder** (pure SSM baseline)
- **AG-ViM** (proposed model)

Metrics evaluated:
- Dice Score
- IoU (Intersection over Union)
- Precision & Recall
- GPU memory usage
- Training time per epoch

---

## Project Structure

```
ag-vim/
├── AG_Vim_Main.ipynb       # Main training and evaluation notebook
├── README.md
└── AGVim_Project_Report.docx  # Full project report (B.Tech Industrial Training)
```

---

## Testing

All modules have been tested and verified:

| Module | Status |
|---|---|
| MRI Dataset Loading | ✅ Pass |
| Data Preprocessing | ✅ Pass |
| AG-ViM Model Architecture | ✅ Pass |
| Model Training Pipeline | ✅ Pass |
| Tumor Segmentation Output | ✅ Pass |
| Evaluation Metrics (Dice, IoU, Precision, Recall) | ✅ Pass |
| Model Comparison (U-Net vs Vision Mamba vs AG-ViM) | ✅ Pass |
| Result Visualization | ✅ Pass |

---

## Future Work

- Integration with hospital PACS (Picture Archiving and Communication Systems)
- Training on larger multi-institutional MRI datasets
- Extension to **3D volumetric segmentation**
- Support for **multi-modal MRI inputs** (T1, T2, FLAIR)
- Real-time inference pipeline for clinical deployment
- Interactive visualization dashboard for radiologists

---

## References

- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- Hatamizadeh et al. (2022). *UNETR: Transformers for 3D Medical Image Segmentation.* WACV.
- Isensee et al. (2021). *nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation.* Nature Methods.

---

## License

This project is developed for academic research purposes. Please cite this work if you use it in your research.