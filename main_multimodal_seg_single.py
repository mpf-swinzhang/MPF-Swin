# --------------------------------------------------------
# Multi-modal Multi-scale Segmentation Training (Single GPU)
# Supports 10-channel optical + 3-channel SAR data
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['LOCAL_RANK'] = '0'

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from config import get_config
from models import build_model

# Global logger (will be initialized in main)
logger = None
from data.build_multimodal_seg import build_loader_multimodal_seg
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper
from timm.utils import AverageMeter

PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


def parse_option():
    parser = argparse.ArgumentParser('Multi-modal Segmentation training script (Single GPU)', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument('--opts', help="Modify config options", default=None, nargs='+')
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained', type=str, help='path to pretrained model')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    args, unparsed = parser.parse_known_args()
    from config import get_config
    config = get_config(args)
    return args, config


def compute_iou(pred, target, num_classes):
    """Compute IoU for each class."""
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    return np.array(ious)


def compute_metrics(pred, target, num_classes):
    """Compute precision, recall, F1 for each class."""
    precisions = []
    recalls = []
    f1_scores = []
    
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        tp = np.logical_and(pred_cls, target_cls).sum()
        fp = np.logical_and(pred_cls, ~target_cls).sum()
        fn = np.logical_and(~pred_cls, target_cls).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1 = 2 * (precision * recall) / (precision + recall) if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0 else float('nan')
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    return np.array(precisions), np.array(recalls), np.array(f1_scores)


def compute_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    return (pred == target).sum() / target.size


def compute_miou(pred, target, num_classes):
    """Compute mean IoU."""
    ious = compute_iou(pred, target, num_classes)
    valid_ious = ious[~np.isnan(ious)]
    if len(valid_ious) == 0:
        return 0.0
    return valid_ious.mean()


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.defrost()
    config.LOCAL_RANK = 0
    config.freeze()
    
    # Build multimodal data loaders
    data_loader_train, data_loader_val, num_classes = build_loader_multimodal_seg(config)
    config.defrost()
    config.MODEL.NUM_CLASSES = num_classes
    config.freeze()
    
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_segmentation=True)
    logger.info(str(model))
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    
    model = model.to(device)
    model_without_ddp = model
    
    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()
    
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    
    from models.seg_losses import balanced_cross_entropy, combined_loss, combined_loss_with_focal, combined_loss_with_edge
    
    # Get loss function type from config
    loss_type = getattr(config.MODEL, 'LOSS_TYPE', 'balanced_ce')  # 'balanced_ce', 'combined', 'combined_focal', 'combined_edge'
    ce_weight = getattr(config.MODEL, 'LOSS_CE_WEIGHT', 0.4)
    dice_weight = getattr(config.MODEL, 'LOSS_DICE_WEIGHT', 0.4)
    focal_weight = getattr(config.MODEL, 'LOSS_FOCAL_WEIGHT', 0.15)
    edge_weight = getattr(config.MODEL, 'LOSS_EDGE_WEIGHT', 0.05)
    
    if loss_type == 'combined':
        logger.info(f"Using Combined Loss (CE: {ce_weight}, Dice: {dice_weight})")
        def criterion(output, target):
            return combined_loss(output, target, ce_weight=ce_weight, dice_weight=dice_weight, ignore_index=255)
    elif loss_type == 'combined_focal':
        logger.info(f"Using Combined Loss with Focal (CE: {ce_weight}, Dice: {dice_weight}, Focal: {focal_weight})")
        def criterion(output, target):
            return combined_loss_with_focal(output, target, ce_weight=ce_weight, dice_weight=dice_weight, 
                                          focal_weight=focal_weight, ignore_index=255)
    elif loss_type == 'combined_edge':
        logger.info(f"Using Combined Loss with Edge (CE: {ce_weight}, Dice: {dice_weight}, Focal: {focal_weight}, Edge: {edge_weight})")
        def criterion(output, target):
            return combined_loss_with_edge(output, target, ce_weight=ce_weight, dice_weight=dice_weight, 
                                          focal_weight=focal_weight, edge_weight=edge_weight, ignore_index=255)
    else:  # 'balanced_ce' or default
        logger.info("Using Balanced Cross-Entropy Loss")
        def criterion(output, target):
            return balanced_cross_entropy(output, target, ignore_index=255)
    
    max_miou = 0.0
    
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
            checkpoint = torch.load(resume_file, map_location='cpu', weights_only=False)
            if 'epoch' in checkpoint:
                if checkpoint['epoch'] + 1 >= config.TRAIN.EPOCHS:
                    logger.warning(f"Training appears to be complete")
    
    if config.MODEL.RESUME:
        max_miou = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        save_imgs = config.EVAL_MODE
        miou, loss = validate_multimodal_seg(config, data_loader_val, model, criterion, device, 
                                           save_images=save_imgs, max_save_images=20, logger=logger)
        logger.info(f"mIoU of the network on test images: {miou:.4f}")
        if config.EVAL_MODE:
            return
    
    if config.MODEL.MULTIMODAL and (config.MODEL.OPTICAL_PRETRAINED or config.MODEL.SAR_PRETRAINED) and (not config.MODEL.RESUME):
        # Load separate pretrained weights for multi-modal model
        from utils import load_multimodal_pretrained
        load_multimodal_pretrained(config, model_without_ddp, logger)
        miou, loss = validate_multimodal_seg(config, data_loader_val, model, criterion, device, 
                                           save_images=config.EVAL_MODE, max_save_images=20, logger=logger)
        logger.info(f"mIoU of the network on test images: {miou:.4f}")
    elif config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        # Fallback to standard pretrained loading
        from utils import load_pretrained
        load_pretrained(config, model_without_ddp, logger)
        miou, loss = validate_multimodal_seg(config, data_loader_val, model, criterion, device, 
                                           save_images=config.EVAL_MODE, max_save_images=20, logger=logger)
        logger.info(f"mIoU of the network on test images: {miou:.4f}")
    
    if config.THROUGHPUT_MODE:
        return
    
    logger.info("Start training")
    logger.info(f"Training from epoch {config.TRAIN.START_EPOCH} to {config.TRAIN.EPOCHS}")
    
    if config.TRAIN.START_EPOCH >= config.TRAIN.EPOCHS:
        logger.warning(f"Training already complete!")
        return
    
    start_time = time.time()
    best_epoch = config.TRAIN.START_EPOCH if config.TRAIN.START_EPOCH > 0 else 0
    
    # Early stopping parameters
    early_stop_patience = getattr(config.TRAIN, 'EARLY_STOP_PATIENCE', 10)  # Stop if no improvement for 10 epochs
    early_stop_min_delta = getattr(config.TRAIN, 'EARLY_STOP_MIN_DELTA', 0.001)  # Minimum improvement threshold
    no_improve_count = 0
    early_stopped = False
    
    # Save multiple checkpoints
    save_freq = getattr(config.TRAIN, 'SAVE_FREQ', 10)  # Save checkpoint every 10 epochs
    keep_last_n = getattr(config.TRAIN, 'KEEP_LAST_N_CHECKPOINTS', 5)  # Keep last 5 checkpoints
    
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch_multimodal_seg(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler,
                                      loss_scaler, device, logger=logger)
        
        save_imgs = (epoch == config.TRAIN.EPOCHS - 1)
        miou, loss = validate_multimodal_seg(config, data_loader_val, model, criterion, device, 
                                           save_images=save_imgs, max_save_images=20, logger=logger, epoch=epoch)
        logger.info(f"mIoU of the network on test images: {miou:.4f}")
        
        # Check for training collapse (sudden drop in performance)
        if epoch > 0 and miou < 0.5 and max_miou > 0.8:
            logger.warning(f"WARNING: Training collapse detected at epoch {epoch}!")
            logger.warning(f"  Current mIoU: {miou:.4f}, Previous best: {max_miou:.4f}")
            logger.warning(f"  This may indicate numerical instability or learning rate issues.")
            # Save current state before potential crash
            emergency_path = os.path.join(config.OUTPUT, f'emergency_epoch_{epoch}.pth')
            try:
                save_checkpoint(config, epoch, model_without_ddp, miou, optimizer, lr_scheduler, loss_scaler,
                              logger, save_path=emergency_path)
                logger.info(f"Emergency checkpoint saved to {emergency_path}")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")
        
        is_best = miou > max_miou
        if is_best:
            max_miou = miou
            best_epoch = epoch
            no_improve_count = 0  # Reset counter
            best_path = os.path.join(config.OUTPUT, 'best_model.pth')
            logger.info(f"New best mIoU: {max_miou:.5f} at epoch {epoch}, saving to {best_path}")
            try:
                # Save with actual epoch number, not 'best' string
                save_checkpoint(config, epoch, model_without_ddp, max_miou, optimizer, lr_scheduler, loss_scaler,
                                logger, save_path=best_path)
            except Exception as e:
                logger.error(f"Failed to save best model at epoch {epoch}: {e}")
        else:
            # Check if improvement is significant
            if miou < max_miou - early_stop_min_delta:
                no_improve_count += 1
            else:
                no_improve_count = 0  # Reset if improvement is within threshold
        
        logger.info(f'Max mIoU: {max_miou:.5f} (at epoch {best_epoch}), No improvement: {no_improve_count}/{early_stop_patience}')
        
        # Save periodic checkpoints
        if (epoch + 1) % save_freq == 0 or epoch == config.TRAIN.EPOCHS - 1:
            ckpt_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
            try:
                save_checkpoint(config, epoch, model_without_ddp, miou, optimizer, lr_scheduler, loss_scaler,
                              logger, save_path=ckpt_path)
                logger.info(f"Periodic checkpoint saved to {ckpt_path}")
                
                # Clean up old checkpoints (keep only last N)
                # Only clean up if we have more than keep_last_n checkpoints
                import glob
                all_ckpts = sorted(glob.glob(os.path.join(config.OUTPUT, 'ckpt_epoch_*.pth')), 
                                 key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
                if len(all_ckpts) > keep_last_n:
                    # Keep the last N checkpoints (including the one we just saved)
                    checkpoints_to_remove = all_ckpts[:-keep_last_n]
                    for old_ckpt in checkpoints_to_remove:
                        try:
                            os.remove(old_ckpt)
                            logger.info(f"Removed old checkpoint: {os.path.basename(old_ckpt)}")
                        except Exception as e:
                            logger.warning(f"Failed to remove old checkpoint {old_ckpt}: {e}")
            except Exception as e:
                logger.error(f"Failed to save periodic checkpoint at epoch {epoch}: {e}")
        
        # Early stopping
        if no_improve_count >= early_stop_patience:
            logger.info(f"Early stopping triggered: No improvement for {early_stop_patience} epochs")
            logger.info(f"Best mIoU: {max_miou:.5f} at epoch {best_epoch}")
            early_stopped = True
            break
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch_multimodal_seg(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, loss_scaler, device, logger=None):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    for idx, ((optical, sar), targets) in enumerate(data_loader):
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(optical, sar)  # Multi-modal forward
            loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]
        
        torch.cuda.synchronize()
        
        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def validate_multimodal_seg(config, data_loader, model, criterion, device, save_images=False, max_save_images=20, logger=None, epoch=0):
    """Validate multi-modal segmentation model."""
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    miou_meter = AverageMeter()
    
    num_classes = config.MODEL.NUM_CLASSES
    all_ious = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    save_count = 0
    vis_dir = os.path.join(config.OUTPUT, 'visualizations')
    if save_images:
        os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, ((optical, sar), targets) in enumerate(data_loader):
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                outputs = model(optical, sar)  # Multi-modal forward
                loss = criterion(outputs, targets)
            
            # Get predictions
            pred = outputs.argmax(dim=1)  # (B, H, W)
            
            # Compute metrics
            batch_size = targets.size(0)
            for b in range(batch_size):
                pred_b = pred[b]
                target_b = targets[b]
                
                acc = compute_accuracy(pred_b, target_b)
                miou = compute_miou(pred_b, target_b, num_classes)
                precisions, recalls, f1s = compute_metrics(pred_b, target_b, num_classes)
                
                acc_meter.update(acc, 1)
                miou_meter.update(miou, 1)
                all_ious.append(compute_iou(pred_b, target_b, num_classes))
                all_precisions.append(precisions)
                all_recalls.append(recalls)
                all_f1s.append(f1s)
            
            loss_meter.update(loss.item(), batch_size)
            
            # Save visualization
            if save_images and save_count < max_save_images:
                for b in range(min(batch_size, max_save_images - save_count)):
                    save_visualization_multimodal(optical[b], sar[b], targets[b], pred[b], 
                                                 vis_dir, save_count, epoch)
                    save_count += 1
    
    # Aggregate metrics
    all_ious = np.array(all_ious)
    all_precisions = np.array(all_precisions)
    all_recalls = np.array(all_recalls)
    all_f1s = np.array(all_f1s)
    
    mean_ious = np.nanmean(all_ious, axis=0)
    mean_precisions = np.nanmean(all_precisions, axis=0)
    mean_recalls = np.nanmean(all_recalls, axis=0)
    mean_f1s = np.nanmean(all_f1s, axis=0)
    
    # Log results
    results = {
        'acc': acc_meter.avg,
        'miou': miou_meter.avg,
        'mf1': np.nanmean(mean_f1s),
    }
    for cls in range(num_classes):
        results[f'iou_{cls}'] = mean_ious[cls] if not np.isnan(mean_ious[cls]) else 0.0
        results[f'F1_{cls}'] = mean_f1s[cls] if not np.isnan(mean_f1s[cls]) else 0.0
        results[f'precision_{cls}'] = mean_precisions[cls] if not np.isnan(mean_precisions[cls]) else 0.0
        results[f'recall_{cls}'] = mean_recalls[cls] if not np.isnan(mean_recalls[cls]) else 0.0
    
    # Save to eval_results.txt
    eval_file = os.path.join(config.OUTPUT, 'eval_results.txt')
    with open(eval_file, 'a', encoding='utf-8') as f:
        f.write(f"Epoch {epoch}:\n")
        for k, v in results.items():
            f.write(f"  {k}: {v:.5f}\n")
        f.write("\n")
    
    if logger:
        logger.info(f"Validation Results (Epoch {epoch}):")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.5f}")
    
    return miou_meter.avg, loss_meter.avg


def save_visualization_multimodal(optical, sar, target, pred, vis_dir, idx, epoch):
    """Save visualization for multi-modal segmentation."""
    # Convert tensors to numpy
    if isinstance(optical, torch.Tensor):
        optical = optical.cpu().numpy()
    if isinstance(sar, torch.Tensor):
        sar = sar.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    # Ensure target and pred are 2D
    if len(target.shape) > 2:
        target = target.squeeze()
    if len(pred.shape) > 2:
        pred = pred.squeeze()
    
    # Handle label format: convert 255 to 1 if present
    target_unique = np.unique(target)
    if 255 in target_unique:
        # Labels are 0 and 255, convert 255 to 1
        target = np.where(target == 255, 1, target)
    
    # Ensure labels are in [0, 1] range
    target = np.clip(target, 0, 1).astype(np.uint8)
    pred = np.clip(pred, 0, 1).astype(np.uint8)
    
    # Denormalize optical (first 3 channels for visualization)
    optical_vis = optical[:3] if len(optical.shape) == 3 else optical[0, :3]
    optical_vis = np.transpose(optical_vis, (1, 2, 0))
    optical_vis = (optical_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    optical_vis = np.clip(optical_vis, 0, 255).astype(np.uint8)
    
    # Denormalize SAR
    sar_vis = np.transpose(sar, (1, 2, 0)) if len(sar.shape) == 3 else sar[0]
    sar_vis = ((sar_vis + 1) / 2 * 255).astype(np.uint8)
    if sar_vis.shape[2] == 1:
        sar_vis = np.repeat(sar_vis, 3, axis=2)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(optical_vis)
    axes[0, 0].set_title('Optical (RGB)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sar_vis)
    axes[0, 1].set_title('SAR')
    axes[0, 1].axis('off')
    
    # GT: white foreground, black background
    # Handle both 0/1 and 0/255 formats
    gt_vis = np.zeros((*target.shape, 3), dtype=np.uint8)
    gt_vis[target == 1] = [255, 255, 255]  # Foreground = white
    axes[1, 0].imshow(gt_vis)
    axes[1, 0].set_title(f'Ground Truth (unique: {np.unique(target)})')
    axes[1, 0].axis('off')
    
    # Prediction: white foreground, black background
    pred_vis = np.zeros((*pred.shape, 3), dtype=np.uint8)
    pred_vis[pred == 1] = [255, 255, 255]  # Foreground = white
    axes[1, 1].imshow(pred_vis)
    axes[1, 1].set_title(f'Prediction (unique: {np.unique(pred)}, class1_pixels: {(pred==1).sum()})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    # Save combined image with DPI 300
    combined_path = os.path.join(vis_dir, f'epoch_{epoch}_sample_{idx}_combined.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual images with DPI 300 using matplotlib
    # Save optical image
    optical_path = os.path.join(vis_dir, f'epoch_{epoch}_sample_{idx}_optical.png')
    fig_opt = plt.figure(figsize=(optical_vis.shape[1]/300, optical_vis.shape[0]/300), dpi=300)
    plt.imshow(optical_vis)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(optical_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig_opt)
    
    # Save SAR image
    sar_path = os.path.join(vis_dir, f'epoch_{epoch}_sample_{idx}_sar.png')
    fig_sar = plt.figure(figsize=(sar_vis.shape[1]/300, sar_vis.shape[0]/300), dpi=300)
    plt.imshow(sar_vis)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(sar_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig_sar)
    
    # Save ground truth
    gt_path = os.path.join(vis_dir, f'epoch_{epoch}_sample_{idx}_gt.png')
    fig_gt = plt.figure(figsize=(gt_vis.shape[1]/300, gt_vis.shape[0]/300), dpi=300)
    plt.imshow(gt_vis)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(gt_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig_gt)
    
    # Save prediction
    pred_path = os.path.join(vis_dir, f'epoch_{epoch}_sample_{idx}_pred.png')
    fig_pred = plt.figure(figsize=(pred_vis.shape[1]/300, pred_vis.shape[0]/300), dpi=300)
    plt.imshow(pred_vis)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(pred_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig_pred)


if __name__ == '__main__':
    args, config = parse_option()
    
    seed = config.SEED
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()
    
    os.makedirs(config.OUTPUT, exist_ok=True)
    # Initialize logger (module-level variable)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")
    # Update module-level logger
    import sys
    sys.modules[__name__].logger = logger
    
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    
    logger.info(f"Full config saved to {path}")
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))
    
    main(config)

