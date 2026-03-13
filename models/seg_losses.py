# --------------------------------------------------------
# Segmentation Loss Functions
# Adapted from BIT_CD-master for better handling of class imbalance
# --------------------------------------------------------

import torch
import torch.nn.functional as F


def balanced_cross_entropy(input, target, weight=None, ignore_index=255):
    """
    类别均衡的交叉熵损失，用于处理类别不平衡问题
    专门针对2类分割任务优化
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*H*W
    :param weight: torch.Tensor, C (未使用，保持接口一致性)
    :param ignore_index: int, 忽略的标签值
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1] or input.shape[-2] != target.shape[-2]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    # 计算正负样本数量
    pos = (target == 1).float()
    neg = (target == 0).float()
    pos_num = torch.sum(pos) + 1e-7
    neg_num = torch.sum(neg) + 1e-7
    total = pos_num + neg_num

    # 计算类别权重：反频率加权（更激进的权重）
    # 对于极度不平衡的数据（如0.98%前景），需要更激进的权重
    # 使用平方根或对数缩放来避免权重过大导致训练不稳定
    weight_neg = 1.0  # 背景权重设为1
    
    # 计算基础权重比
    base_ratio = neg_num / pos_num  # 约等于 99.02/0.98 ≈ 101
    
    # 使用更激进的权重策略：对极度不平衡的情况，使用平方根缩放
    # 这样可以获得更高的权重但不会过大（sqrt(101) ≈ 10）
    # 或者使用对数缩放：log(1 + ratio) 会更温和
    # 这里使用 sqrt 缩放，对于极度不平衡的情况更有效
    if base_ratio > 50:
        # 极度不平衡：使用 sqrt 缩放
        weight_pos = torch.sqrt(base_ratio) * 2.0  # 约 10 * 2 = 20
    elif base_ratio > 10:
        # 中度不平衡：使用 sqrt 缩放
        weight_pos = torch.sqrt(base_ratio) * 1.5
    else:
        # 轻度不平衡：直接使用比例
        weight_pos = base_ratio
    
    # 创建类别权重 [背景权重, 前景权重]
    class_weights = torch.tensor([weight_neg, weight_pos.item()], device=input.device, dtype=input.dtype)
    
    # 归一化权重（可选，但通常不需要）
    # class_weights = class_weights / class_weights.sum() * 2.0
    
    # 使用加权交叉熵
    loss = F.cross_entropy(input=input, target=target, weight=class_weights, 
                           ignore_index=ignore_index, reduction='mean')
    
    return loss


def focal_loss(input, target, alpha=0.25, gamma=2.0, ignore_index=255):
    """
    Focal Loss: 专门用于处理类别不平衡和小目标检测
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*H*W
    :param alpha: float, 平衡因子，通常为0.25
    :param gamma: float, 聚焦参数，通常为2.0，值越大对难样本关注越多
    :param ignore_index: int, 忽略的标签值
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1] or input.shape[-2] != target.shape[-2]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)
    
    # 计算交叉熵
    ce_loss = F.cross_entropy(input, target, reduction='none', ignore_index=ignore_index)
    
    # 计算概率
    pt = torch.exp(-ce_loss)
    
    # 计算alpha权重（为每个类别设置不同的alpha）
    # 对于极度不平衡的数据，给前景更高的alpha
    pos = (target == 1).float()
    neg = (target == 0).float()
    pos_num = torch.sum(pos) + 1e-7
    neg_num = torch.sum(neg) + 1e-7
    
    # 动态计算alpha：前景使用更高的alpha
    alpha_t = alpha * pos + (1 - alpha) * neg
    # 对于极度不平衡的情况，可以进一步调整alpha
    if pos_num / (pos_num + neg_num) < 0.01:  # 前景占比 < 1%
        alpha_t = alpha_t * 2.0  # 进一步增加前景权重
    
    # Focal Loss
    focal_loss = alpha_t * (1 - pt) ** gamma * ce_loss
    
    return focal_loss.mean()


def dice_loss(input, target, smooth=1e-5, ignore_index=255):
    """
    Dice Loss for segmentation tasks.
    Dice Loss = 1 - Dice Coefficient
    
    :param input: torch.Tensor, N*C*H*W, logits
    :param target: torch.Tensor, N*H*W, class indices
    :param smooth: float, smoothing factor to avoid division by zero
    :param ignore_index: int, ignore label value
    :return: torch.Tensor [0] dice loss
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1] or input.shape[-2] != target.shape[-2]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)
    
    # Get probabilities
    probs = F.softmax(input, dim=1)  # (N, C, H, W)
    
    # Create one-hot encoding for target
    num_classes = input.shape[1]
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()  # (N, C, H, W)
    
    # Mask out ignore_index
    if ignore_index is not None:
        mask = (target != ignore_index).float()
        mask = mask.unsqueeze(1)  # (N, 1, H, W)
        probs = probs * mask
        target_one_hot = target_one_hot * mask
    
    # Calculate Dice coefficient for each class
    intersection = (probs * target_one_hot).sum(dim=(2, 3))  # (N, C)
    union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # (N, C)
    
    # Dice coefficient: 2 * intersection / union
    dice = (2.0 * intersection + smooth) / (union + smooth)  # (N, C)
    
    # Dice loss: 1 - dice (average over classes and batch)
    dice_loss = 1.0 - dice.mean()
    
    return dice_loss


def combined_loss(input, target, ce_weight=0.5, dice_weight=0.5, ignore_index=255):
    """
    Combined Loss: Balanced Cross-Entropy + Dice Loss
    
    This combination leverages:
    - Cross-Entropy: Good gradient properties, handles class imbalance
    - Dice Loss: Directly optimizes IoU, good for segmentation tasks
    
    :param input: torch.Tensor, N*C*H*W, logits
    :param target: torch.Tensor, N*H*W, class indices
    :param ce_weight: float, weight for cross-entropy loss (default 0.5)
    :param dice_weight: float, weight for dice loss (default 0.5)
    :param ignore_index: int, ignore label value
    :return: torch.Tensor [0] combined loss
    """
    # Balanced Cross-Entropy Loss
    ce_loss = balanced_cross_entropy(input, target, ignore_index=ignore_index)
    
    # Dice Loss
    dice_loss_val = dice_loss(input, target, ignore_index=ignore_index)
    
    # Combined loss
    total_loss = ce_weight * ce_loss + dice_weight * dice_loss_val
    
    return total_loss


def combined_loss_with_focal(input, target, ce_weight=0.4, dice_weight=0.4, focal_weight=0.2, 
                             ignore_index=255, alpha=0.25, gamma=2.0):
    """
    Combined Loss: Balanced Cross-Entropy + Dice Loss + Focal Loss
    
    This combination leverages:
    - Cross-Entropy: Good gradient properties
    - Dice Loss: Directly optimizes IoU
    - Focal Loss: Focuses on hard examples
    
    :param input: torch.Tensor, N*C*H*W, logits
    :param target: torch.Tensor, N*H*W, class indices
    :param ce_weight: float, weight for cross-entropy loss
    :param dice_weight: float, weight for dice loss
    :param focal_weight: float, weight for focal loss
    :param ignore_index: int, ignore label value
    :param alpha: float, focal loss alpha parameter
    :param gamma: float, focal loss gamma parameter
    :return: torch.Tensor [0] combined loss
    """
    # Balanced Cross-Entropy Loss
    ce_loss = balanced_cross_entropy(input, target, ignore_index=ignore_index)
    
    # Dice Loss
    dice_loss_val = dice_loss(input, target, ignore_index=ignore_index)
    
    # Focal Loss
    focal_loss_val = focal_loss(input, target, alpha=alpha, gamma=gamma, ignore_index=ignore_index)
    
    # Combined loss
    total_loss = ce_weight * ce_loss + dice_weight * dice_loss_val + focal_weight * focal_loss_val
    
    return total_loss


def edge_loss(input, target, ignore_index=255):
    """
    Edge Loss: Focuses on boundary/edge regions for better detail preservation.
    
    Uses Sobel operator to extract edges from both prediction and ground truth,
    then computes loss on edge regions.
    
    :param input: torch.Tensor, N*C*H*W, logits
    :param target: torch.Tensor, N*H*W, class indices
    :param ignore_index: int, ignore label value
    :return: torch.Tensor [0] edge loss
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1] or input.shape[-2] != target.shape[-2]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)
    
    # Get prediction probabilities
    probs = F.softmax(input, dim=1)  # (N, C, H, W)
    pred = probs.argmax(dim=1).float()  # (N, H, W)
    target_float = target.float()  # (N, H, W)
    
    # Create Sobel kernels for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=input.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=input.device).view(1, 1, 3, 3)
    
    # Extract edges from prediction
    pred_edges_x = F.conv2d(pred.unsqueeze(1), sobel_x, padding=1)  # (N, 1, H, W)
    pred_edges_y = F.conv2d(pred.unsqueeze(1), sobel_y, padding=1)  # (N, 1, H, W)
    pred_edges = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2 + 1e-6)  # (N, 1, H, W)
    
    # Extract edges from ground truth
    target_edges_x = F.conv2d(target_float.unsqueeze(1), sobel_x, padding=1)  # (N, 1, H, W)
    target_edges_y = F.conv2d(target_float.unsqueeze(1), sobel_y, padding=1)  # (N, 1, H, W)
    target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2 + 1e-6)  # (N, 1, H, W)
    
    # Normalize edges to [0, 1]
    pred_edges = pred_edges / (pred_edges.max() + 1e-6)
    target_edges = target_edges / (target_edges.max() + 1e-6)
    
    # Mask out ignore_index regions
    if ignore_index is not None:
        mask = (target != ignore_index).float().unsqueeze(1)  # (N, 1, H, W)
        pred_edges = pred_edges * mask
        target_edges = target_edges * mask
    
    # Compute L1 loss on edges
    edge_loss = F.l1_loss(pred_edges, target_edges, reduction='mean')
    
    return edge_loss


def combined_loss_with_edge(input, target, ce_weight=0.4, dice_weight=0.4, focal_weight=0.15, 
                           edge_weight=0.05, ignore_index=255, alpha=0.25, gamma=2.0):
    """
    Combined Loss: Balanced Cross-Entropy + Dice Loss + Focal Loss + Edge Loss
    
    This combination leverages:
    - Cross-Entropy: Good gradient properties
    - Dice Loss: Directly optimizes IoU
    - Focal Loss: Focuses on hard examples
    - Edge Loss: Improves boundary/detail accuracy
    
    :param input: torch.Tensor, N*C*H*W, logits
    :param target: torch.Tensor, N*H*W, class indices
    :param ce_weight: float, weight for cross-entropy loss
    :param dice_weight: float, weight for dice loss
    :param focal_weight: float, weight for focal loss
    :param edge_weight: float, weight for edge loss
    :param ignore_index: int, ignore label value
    :param alpha: float, focal loss alpha parameter
    :param gamma: float, focal loss gamma parameter
    :return: torch.Tensor [0] combined loss
    """
    # Balanced Cross-Entropy Loss
    ce_loss = balanced_cross_entropy(input, target, ignore_index=ignore_index)
    
    # Dice Loss
    dice_loss_val = dice_loss(input, target, ignore_index=ignore_index)
    
    # Focal Loss
    focal_loss_val = focal_loss(input, target, alpha=alpha, gamma=gamma, ignore_index=ignore_index)
    
    # Edge Loss
    edge_loss_val = edge_loss(input, target, ignore_index=ignore_index)
    
    # Combined loss
    total_loss = (ce_weight * ce_loss + 
                 dice_weight * dice_loss_val + 
                 focal_weight * focal_loss_val + 
                 edge_weight * edge_loss_val)
    
    return total_loss

