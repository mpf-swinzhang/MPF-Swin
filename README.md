# MPF-Swin: Multi-Path Fusion Swin (Optical 10-channel + SAR 3-channel Multimodal Segmentation)

This repository provides training and evaluation code for **MPF-Swin**: a multimodal semantic segmentation framework based on Swin Transformer / SwinV2, designed for **optical (10-channel) + SAR (3-channel)** fusion segmentation tasks.

## Environment Setup

Python 3.9+ is recommended. First install PyTorch that matches your CUDA version, then:

```bash
pip install -r requirements.txt
```

## Dataset Directory Structure

This public repository **does not contain any data**. Multimodal segmentation data are expected to be loaded from `--data-path` (or config `DATA.DATA_PATH`) with the following structure:

```text
<DATA_ROOT>/
  A_opt/                 # optical images (10-channel), *.tif
  B_sar/                 # SAR images (3-channel), *.tif
  label/                 # segmentation labels, *.tif
  list/
    train.txt            # one sample ID per line (without file extension)
    val.txt
```

Note: sample IDs in `train.txt` / `val.txt` are used to compose corresponding `.tif` file paths under `A_opt/`, `B_sar/`, and `label/`.

## Quick Start (Recommended: Windows, Single GPU)

### Training (Multimodal Segmentation, Single GPU)

```bash
python main_multimodal_seg_single.py ^
  --cfg configs\swinv2\swinv2_base_patch4_window8_256_multimodal_seg.yaml ^
  --data-path <DATA_ROOT> ^
  --output output
```

### Validation / Inference (from Checkpoint)

```bash
python main_multimodal_seg_single.py ^
  --cfg configs\swinv2\swinv2_base_patch4_window8_256_multimodal_seg.yaml ^
  --data-path <DATA_ROOT> ^
  --resume <PATH_TO_CKPT.pth> ^
  --eval
```

## Configuration & Options

Main config file:

- `configs/swinv2/swinv2_base_patch4_window8_256_multimodal_seg.yaml`

Common options (all under `MODEL.*` in the config file):

- `USE_STAGE_FUSION`: enable **MPF (stage-level fusion)** or not
- `USE_BOUNDARY_ENHANCEMENT`: boundary enhancement
- `USE_DETAIL_ENHANCEMENT`: detail enhancement
- `LOSS_TYPE`: `balanced_ce` / `combined` / `combined_focal` / `combined_edge`

## Pretrained Weights (Optional)

This repository **does not ship with any `.pth/.ckpt` weight files**. To use pretrained weights, set:

- `MODEL.OPTICAL_PRETRAINED`
- `MODEL.SAR_PRETRAINED`

If not set, training from scratch is also supported.

## Directory Layout

```text
configs/         # YAML configs
data/            # data loading and augmentation
kernels/         # optional CUDA/window kernels
models/          # MPF-Swin model implementations
main_*.py        # training / evaluation entry points
utils*.py logger.py optimizer.py lr_scheduler.py config.py
```

## License

See `LICENSE`.

