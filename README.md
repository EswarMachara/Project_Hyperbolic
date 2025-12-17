# Project Hyperbolic

Research workspace for:
- **HANS-Net** (Hyperbolic Attention Network for Segmentation) - hyperbolic geometry + attention + wavelets for CT slice segmentation.
- **CDBNs / ConvRBMs** — Convolutional Deep Belief Networks (stack of RBMs) for OCT image classification.

## Repository layout

- `HANSNet/`
  - `HANSNET.pdf` - paper reference.
  - `model.py` - full PyTorch implementation of HANS-Net.
  - `dataloader.py` - LiTS-style slice dataset parsing + loaders.
  - `trainer.py` - training loop + checkpointing + CLI.
  - `model.ipynb` - component-by-component notebook (building blocks).
  - `run_all_pipeline.ipynb` — notebook pipeline (Kaggle-ready, end-to-end).

- `OCT_CDBNs/`
  - `CDBN_OCT_Classification.ipynb` — Kaggle-oriented CDBN pipeline.
  - `CDBN_OCT_LocalGPU_5GB.ipynb` — local-GPU optimized variant (large dataset, memory safety).

---

## HANS-Net (segmentation)

<p align="center">
  <img src="figures\HANSNet.jpg" width="700">
</p>

### What it is
`HANSNet/model.py` implements a U-Net style segmentation model with:
- **Wavelet decomposition (Haar DWT)** for multi-frequency features
- **Synaptic plasticity blocks** (Hebbian-inspired feature modulation)
- **Temporal attention** over **3 consecutive slices** (T=3)
- **Hyperbolic convolutions** in the bottleneck (Poincaré ball geometry)
- **INR branch** for boundary refinement

**Input:** `[B, 3, 1, H, W]` (3 grayscale slices)

**Output:** `[B, 1, H, W]` (logits for the center slice)

### Dataset expectations
The dataloader assumes a LiTS-like *slice* layout:

```
Dataset/
  train_images/
    train_images/
      volume-<case>_slice_<number>.jpg
  train_masks/
    volume-<case>_slice_<number>.jpg
```

Important constraints enforced by `HANSNet/dataloader.py`:
- Image/mask filenames must match exactly (same basename)
- Slice filenames must contain `_slice_` and match `volume-<number>_slice_<number>.(jpg|png)`
- Missing masks will raise an error during metadata build

### Quickstart (CLI training)
From the repo root (this folder):

```bash
python -m pip install -U torch torchvision pillow tqdm
python HANSNet\trainer.py --data-root Dataset --epochs 3 --batch-size 2 --device cuda
```

Checkpoints:
- `checkpoints/last.pt`
- `checkpoints/best.pt`

### Notebooks
- `HANSNet/model.ipynb` — implements and validates core building blocks.
- `HANSNet/run_all_pipeline.ipynb` — self-contained pipeline (includes Kaggle path detection and training).

### Common issues
- **Inconsistent dataset structure / naming:** the loader is strict by design; fix folder names or filename patterns first.
- **CUDA OOM:** lower `--batch-size`, reduce `img_size` in `build_dataloaders(...)`, or reduce model width via `HANSNet(base_channels=16)`.

---

## OCT CDBNs / ConvRBMs (classification)

### What it is
The notebooks in `OCT_CDBNs/` implement a **Convolutional Deep Belief Network** pipeline (stacked RBMs) for **multi-class OCT image classification**.

Two versions are included:
- `OCT_CDBNs/CDBN_OCT_Classification.ipynb` — Kaggle-focused configuration (GPU required).
- `OCT_CDBNs/CDBN_OCT_LocalGPU_5GB.ipynb` — local GPU version designed for a ~5.5GB dataset, emphasizing memory-safe loading and disk-backed caching.

### Typical requirements
These notebooks use PyTorch + common ML tooling. A minimal base you’ll likely need:

```bash
python -m pip install -U torch torchvision numpy matplotlib
```

(Additional packages may be required depending on which notebook sections you run.)

### Notes on GPU memory (OOM)
If you hit out-of-memory during ConvRBM/CDBN stages, the most effective levers are:
- Reduce **RBM/feature extractor batch sizes** (the local notebook documents stage-specific batch sizes).
- Prefer **streaming/disk-backed caching** instead of caching full latent tensors in RAM/VRAM.
- Reduce input resolution / crop aggressively.
- If available, enable mixed precision (AMP) in the training loops.

---

