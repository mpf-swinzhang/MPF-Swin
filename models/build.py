# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .multimodal_seg import MultiModalSwinV2Seg


def build_model(config, is_pretrain=False, is_segmentation=False):
    """
    Minimal build function for this public repo.

    Supported:
    - Multi-modal semantic segmentation with SwinV2 backbone (`MODEL.TYPE: swinv2`, `MODEL.MULTIMODAL: True`)
    """
    if is_pretrain:
        raise NotImplementedError("This repo variant only keeps multi-modal segmentation.")
    if not is_segmentation:
        raise NotImplementedError("This repo variant only keeps multi-modal segmentation.")

    if not getattr(config.MODEL, "MULTIMODAL", False):
        raise NotImplementedError("Only multi-modal segmentation is supported (MODEL.MULTIMODAL: True).")

    model_type = config.MODEL.TYPE
    if model_type != "swinv2":
        raise NotImplementedError("Only SwinV2 backbone is supported in this trimmed repo (MODEL.TYPE: swinv2).")

    return MultiModalSwinV2Seg(config, num_classes=config.MODEL.NUM_CLASSES)
