# MPF-Swin：Multi-Path Fusion Swin（光学 10 通道 + SAR 3 通道 多模态分割）

本仓库提供 **MPF-Swin** 的训练/验证代码：基于 Swin Transformer / SwinV2 的多模态语义分割框架，面向 **光学（10 通道）+ SAR（3 通道）** 的融合分割任务。

## 环境安装

建议 Python 3.9+。先安装与你 CUDA 匹配的 PyTorch，然后：

```bash
pip install -r requirements.txt
```

## 数据集目录结构

公开仓库 **不包含任何数据**。多模态分割数据默认从 `--data-path`（或配置 `DATA.DATA_PATH`）读取，期望结构如下：

```text
<DATA_ROOT>/
  A_opt/                 # optical images (10-channel), *.tif
  B_sar/                 # SAR images (3-channel), *.tif
  label/                 # segmentation labels, *.tif
  list/
    train.txt            # 每行一个样本 ID（不含扩展名）
    val.txt
```

说明：`train.txt/val.txt` 中的样本 ID 会用于在 `A_opt/ B_sar/ label/` 下拼接找到对应的 `.tif` 文件。

## 快速开始（推荐：Windows 单卡）

### 训练（多模态分割，单 GPU）

```bash
python main_multimodal_seg_single.py ^
  --cfg configs\swinv2\swinv2_base_patch4_window8_256_multimodal_seg.yaml ^
  --data-path <DATA_ROOT> ^
  --output output
```

### 验证/推理（从 checkpoint）

```bash
python main_multimodal_seg_single.py ^
  --cfg configs\swinv2\swinv2_base_patch4_window8_256_multimodal_seg.yaml ^
  --data-path <DATA_ROOT> ^
  --resume <PATH_TO_CKPT.pth> ^
  --eval
```

## 配置与开关

主要配置文件：

- `configs/swinv2/swinv2_base_patch4_window8_256_multimodal_seg.yaml`

常用开关（都在配置文件 `MODEL.*` 下）：

- `USE_STAGE_FUSION`：是否启用 **MPF（stage-level fusion）**
- `USE_BOUNDARY_ENHANCEMENT`：边界增强
- `USE_DETAIL_ENHANCEMENT`：细节增强
- `LOSS_TYPE`：`balanced_ce` / `combined` / `combined_focal` / `combined_edge`

## 预训练权重（可选）

本仓库 **不自带任何 `.pth/.ckpt` 权重文件**。如需使用预训练权重，请在配置中设置：

- `MODEL.OPTICAL_PRETRAINED`
- `MODEL.SAR_PRETRAINED`

不设置也可以从头训练。

## 目录结构

```text
configs/         # YAML 配置
data/            # 数据加载与增强
kernels/         # 可选 CUDA/window kernel
models/          # MPF-Swin 模型实现
main_*.py        # 训练/验证入口
utils*.py logger.py optimizer.py lr_scheduler.py config.py
```

## License

See `LICENSE`.


