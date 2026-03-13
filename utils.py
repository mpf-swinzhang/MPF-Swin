# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist

try:
    from torch._six import inf
except:
    from torch import inf


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=False)
    
    # Handle channel mismatch (e.g., 3-channel checkpoint vs 10-channel model)
    state_dict = checkpoint['model']
    model_state_dict = model.state_dict()
    
    # Filter out incompatible layers (e.g., patch_embed with different input channels)
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                logger.warning(f"Skipping {k} due to shape mismatch: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
        else:
            logger.warning(f"Skipping {k} (not found in model)")
    
    msg = model.load_state_dict(filtered_state_dict, strict=False)
    logger.info(msg)
    logger.info(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} compatible layers from checkpoint")
    
    max_accuracy = 0.0
    
    # Check if there's a channel mismatch that would affect optimizer state
    has_channel_mismatch = False
    for k in state_dict.keys():
        if 'patch_embed.proj.weight' in k or 'backbone.patch_embed.proj.weight' in k:
            if k in model_state_dict:
                if state_dict[k].shape[1] != model_state_dict[k].shape[1]:
                    has_channel_mismatch = True
                    logger.warning(f"Channel mismatch detected in {k}: checkpoint {state_dict[k].shape[1]} vs model {model_state_dict[k].shape[1]} channels")
                    break
    
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        if has_channel_mismatch:
            logger.warning("Skipping optimizer/lr_scheduler state due to channel mismatch - will start training from scratch")
        else:
            # Check if optimizer state is compatible
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("Optimizer state loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}, will continue without it")
            
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                logger.info("LR scheduler state loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load lr_scheduler state: {e}, will continue without it")
        
        config.defrost()
        if 'epoch' in checkpoint and not has_channel_mismatch:
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        else:
            config.TRAIN.START_EPOCH = 0
        config.freeze()
        
        if 'scaler' in checkpoint:
            try:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            except Exception as e:
                logger.warning(f"Failed to load scaler state: {e}")
        
        if 'epoch' in checkpoint and not has_channel_mismatch:
            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        else:
            logger.info(f"=> loaded model weights from '{config.MODEL.RESUME}' (starting from epoch 0)")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    
    # Check if pretrained weight file exists
    if not os.path.exists(config.MODEL.PRETRAINED):
        error_msg = f"""
{'='*80}
错误: 预训练权重文件未找到！
文件路径: {config.MODEL.PRETRAINED}

请按照以下步骤操作：

1. 下载预训练权重文件：
   
   SwinV2 Base (ImageNet-1K):
   - GitHub: https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth
   - 百度网盘: https://pan.baidu.com/s/18AfMSz3dPyzIvP1dKuERvQ?pwd=swin
   
   SwinV2 Base (ImageNet-22K):
   - (upstream reference) swinv2_base_patch4_window12_192_22k.pth
   - 百度网盘: https://pan.baidu.com/s/1Xc2rsSsRQz_sy5mjgfxrMQ?pwd=swin

2. 将下载的 .pth 文件放在项目根目录，或修改训练脚本中的路径

3. 如果不想使用预训练权重，可以从配置文件中移除 --pretrained 参数
{'='*80}
"""
        logger.error(error_msg)
        raise FileNotFoundError(f"预训练权重文件未找到: {config.MODEL.PRETRAINED}\n请查看上面的错误信息获取下载链接。")
    
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    # Skip head processing for segmentation models (they don't have a classification head)
    has_head = hasattr(model, 'head') and model.head is not None
    is_segmentation = hasattr(model, 'seg_head')
    
    if 'head.bias' in state_dict and has_head:
        head_bias_pretrained = state_dict['head.bias']
        Nc1 = head_bias_pretrained.shape[0]
        Nc2 = model.head.bias.shape[0]
        if (Nc1 != Nc2):
            if Nc1 == 21841 and Nc2 == 1000:
                logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
                map22kto1k_path = f'data/map22kto1k.txt'
                with open(map22kto1k_path) as f:
                    map22kto1k = f.readlines()
                map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
                state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
            else:
                torch.nn.init.constant_(model.head.bias, 0.)
                torch.nn.init.constant_(model.head.weight, 0.)
                del state_dict['head.weight']
                del state_dict['head.bias']
                logger.warning(f"Error in loading classifier head, re-init classifier head to 0")
    elif 'head.bias' in state_dict and is_segmentation:
        # For segmentation models, remove head weights from pretrained checkpoint
        logger.info("Segmentation model detected, removing classification head weights from pretrained checkpoint")
        if 'head.weight' in state_dict:
            del state_dict['head.weight']
        if 'head.bias' in state_dict:
            del state_dict['head.bias']
        if 'norm.weight' in state_dict:
            del state_dict['norm.weight']
        if 'norm.bias' in state_dict:
            del state_dict['norm.bias']
    
    # Filter out incompatible layers (e.g., patch_embed with different input channels)
    # This is especially important for segmentation models that may have different input channels
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    skipped_layers = []
    
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                skipped_layers.append(f"{k} (shape mismatch: {v.shape} vs {model_state_dict[k].shape})")
        else:
            # Skip keys that don't exist in model (e.g., head for segmentation models)
            if not (k.startswith('head.') or k.startswith('seg_head.')):
                skipped_layers.append(f"{k} (not in model)")
    
    if skipped_layers:
        logger.info(f"Skipping {len(skipped_layers)} incompatible layers from pretrained checkpoint")
        if len(skipped_layers) <= 10:
            for layer in skipped_layers:
                logger.info(f"  - {layer}")
        else:
            for layer in skipped_layers[:10]:
                logger.info(f"  - {layer}")
            logger.info(f"  ... and {len(skipped_layers) - 10} more layers")
    
    msg = model.load_state_dict(filtered_state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def load_multimodal_pretrained(config, model, logger, optical_pretrained=None, sar_pretrained=None):
    """Load pretrained weights for multi-modal model (separate weights for optical and SAR branches).
    
    Args:
        config: Configuration object
        model: Multi-modal model with optical_backbone and sar_backbone
        logger: Logger object
        optical_pretrained: Path to pretrained weights for optical branch (10 channels)
        sar_pretrained: Path to pretrained weights for SAR branch (3 channels)
    """
    if optical_pretrained is None:
        optical_pretrained = getattr(config.MODEL, 'OPTICAL_PRETRAINED', None)
    if sar_pretrained is None:
        sar_pretrained = getattr(config.MODEL, 'SAR_PRETRAINED', None)
    
    # Load optical branch pretrained weights
    if optical_pretrained and os.path.exists(optical_pretrained):
        logger.info(f"==============> Loading optical branch weights from {optical_pretrained}......")
        checkpoint = torch.load(optical_pretrained, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        
        # Process state dict (remove incompatible keys)
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]
        
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del state_dict[k]
        
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]
        
        # Remove head weights
        head_keys = [k for k in state_dict.keys() if k.startswith('head.') or k.startswith('norm.')]
        for k in head_keys:
            del state_dict[k]
        
        # Adapt 3-channel weights to 10-channel for optical branch
        optical_state_dict = {}
        for k, v in state_dict.items():
            # Map to optical_backbone
            new_key = k.replace('backbone.', '') if 'backbone.' in k else k
            new_key = 'optical_backbone.' + new_key
            
            # Handle patch_embed.proj.weight: adapt from 3 channels to 10 channels
            if 'patch_embed.proj.weight' in k or new_key.endswith('patch_embed.proj.weight'):
                if v.shape[1] == 3 and model.optical_backbone.patch_embed.proj.weight.shape[1] == 10:
                    # Expand 3-channel weights to 10 channels
                    logger.info(f"Adapting patch_embed from 3 channels to 10 channels for optical branch")
                    # Strategy: repeat RGB channels: R, G, B, R, G, B, R, G, B, R
                    expanded_weight = torch.cat([
                        v,  # R, G, B (channels 0-2)
                        v,  # R, G, B (channels 3-5)
                        v[:, :2, :, :],  # R, G (channels 6-7)
                        v[:, :1, :, :],  # R (channel 8)
                        v[:, :1, :, :],  # R (channel 9)
                    ], dim=1)
                    optical_state_dict[new_key] = expanded_weight
                elif v.shape == model.optical_backbone.patch_embed.proj.weight.shape:
                    optical_state_dict[new_key] = v
                else:
                    logger.warning(f"Skipping {new_key} due to shape mismatch: {v.shape} vs {model.optical_backbone.patch_embed.proj.weight.shape}")
            else:
                # Check if key exists in optical_backbone
                if new_key in model.state_dict():
                    if v.shape == model.state_dict()[new_key].shape:
                        optical_state_dict[new_key] = v
                    else:
                        logger.warning(f"Skipping {new_key} due to shape mismatch")
                else:
                    logger.debug(f"Skipping {new_key} (not in optical_backbone)")
        
        msg = model.load_state_dict(optical_state_dict, strict=False)
        logger.info(f"Optical branch: {msg}")
        logger.info(f"Loaded {len(optical_state_dict)} layers for optical branch")
        del checkpoint, state_dict, optical_state_dict
        torch.cuda.empty_cache()
    elif optical_pretrained:
        logger.warning(f"Optical pretrained weights not found: {optical_pretrained}")
    
    # Load SAR branch pretrained weights
    if sar_pretrained and os.path.exists(sar_pretrained):
        logger.info(f"==============> Loading SAR branch weights from {sar_pretrained}......")
        checkpoint = torch.load(sar_pretrained, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        
        # Process state dict (same as optical)
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]
        
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del state_dict[k]
        
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]
        
        # Remove head weights
        head_keys = [k for k in state_dict.keys() if k.startswith('head.') or k.startswith('norm.')]
        for k in head_keys:
            del state_dict[k]
        
        # Map to sar_backbone (3 channels should match directly)
        sar_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('backbone.', '') if 'backbone.' in k else k
            new_key = 'sar_backbone.' + new_key
            
            if new_key in model.state_dict():
                if v.shape == model.state_dict()[new_key].shape:
                    sar_state_dict[new_key] = v
                else:
                    logger.warning(f"Skipping {new_key} due to shape mismatch: {v.shape} vs {model.state_dict()[new_key].shape}")
            else:
                logger.debug(f"Skipping {new_key} (not in sar_backbone)")
        
        msg = model.load_state_dict(sar_state_dict, strict=False)
        logger.info(f"SAR branch: {msg}")
        logger.info(f"Loaded {len(sar_state_dict)} layers for SAR branch")
        del checkpoint, state_dict, sar_state_dict
        torch.cuda.empty_cache()
    elif sar_pretrained:
        logger.warning(f"SAR pretrained weights not found: {sar_pretrained}")
    
    if not optical_pretrained and not sar_pretrained:
        logger.warning("No pretrained weights specified for multi-modal model. Training from scratch.")


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, save_path=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    if save_path is None:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Use temporary file and atomic rename for safer saving
    temp_path = save_path + '.tmp'
    
    try:
        logger.info(f"{save_path} saving......")
        
        # Check available disk space (rough estimate)
        try:
            import shutil
            stat = shutil.disk_usage(os.path.dirname(save_path))
            free_gb = stat.free / (1024**3)
            logger.info(f"Available disk space: {free_gb:.2f} GB")
            if free_gb < 1.0:
                logger.warning(f"Low disk space: {free_gb:.2f} GB. Checkpoint saving may fail.")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        # Save to temporary file first
        torch.save(save_state, temp_path, _use_new_zipfile_serialization=False)
        
        # Atomic rename (works on Windows and Linux)
        if os.path.exists(save_path):
            # On Windows, need to remove old file first
            try:
                os.remove(save_path)
            except PermissionError:
                # File might be in use, try renaming it
                old_backup = save_path + '.old'
                if os.path.exists(old_backup):
                    os.remove(old_backup)
                os.rename(save_path, old_backup)
        
        os.rename(temp_path, save_path)
        logger.info(f"{save_path} saved !!!")
        
    except RuntimeError as e:
        logger.error(f"Failed to save checkpoint to {save_path}: {e}")
        logger.error("This might be due to:")
        logger.error("  1. Insufficient disk space")
        logger.error("  2. File permission issues")
        logger.error("  3. File is being used by another process")
        
        # Try to clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        # Re-raise the exception so caller can handle it
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving checkpoint: {e}")
        # Try to clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
