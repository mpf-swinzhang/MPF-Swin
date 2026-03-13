# --------------------------------------------------------
# Detail Enhancement Module
# For improving fine-grained detail preservation in segmentation
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDetailExtraction(nn.Module):
    """Multi-scale detail extraction using parallel convolutions."""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Multi-scale convolutions for detail extraction
        # 1x1: point details
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 3x3: local details
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 5x5: regional details
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise separable for efficiency
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1, groups=channels // 4),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, kernel_size=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input features
        Returns:
            detail_feat: (B, C, H, W) detail-enhanced features
        """
        # Extract multi-scale details
        d1 = self.conv1x1(x)
        d3 = self.conv3x3(x)
        d5 = self.conv5x5(x)
        dw = self.dw_conv(x)
        
        # Concatenate
        detail = torch.cat([d1, d3, d5, dw], dim=1)  # (B, C, H, W)
        
        # Fusion with residual
        detail = self.fusion(detail)
        detail = detail + x  # Residual connection
        
        return detail


class SpatialAttention(nn.Module):
    """Spatial attention for detail regions."""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input features
        Returns:
            attended: (B, C, H, W) spatially attended features
        """
        # Average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        # Generate attention map
        attn = self.conv(x_cat)  # (B, 1, H, W)
        attn = self.sigmoid(attn)
        
        # Apply attention
        attended = x * attn
        
        return attended


class DetailEnhancementModule(nn.Module):
    """Detail Enhancement Module for improving fine-grained details."""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Multi-scale detail extraction
        self.detail_extraction = MultiScaleDetailExtraction(channels)
        
        # Spatial attention for detail regions
        self.spatial_attn = SpatialAttention(kernel_size=7)
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input features
        Returns:
            enhanced: (B, C, H, W) detail-enhanced features
        """
        # Extract multi-scale details
        detail = self.detail_extraction(x)
        
        # Apply spatial attention
        detail = self.spatial_attn(detail)
        
        # Apply channel attention
        ca = self.channel_attn(detail)
        detail = detail * ca
        
        # Refinement with residual
        enhanced = self.refine(detail)
        enhanced = enhanced + x  # Residual connection
        
        return enhanced


class ProgressiveDetailEnhancement(nn.Module):
    """Progressive Detail Enhancement for FPN decoder.
    
    Applies detail enhancement at different levels with varying strength:
    - Shallow layers (high resolution): Strong detail enhancement
    - Deep layers (low resolution): Moderate detail enhancement
    """
    
    def __init__(self, channels_list, use_detail_enhancement=True):
        """
        Args:
            channels_list: List of channel numbers for each FPN level
            use_detail_enhancement: Whether to use detail enhancement
        """
        super().__init__()
        self.use_detail_enhancement = use_detail_enhancement
        self.num_levels = len(channels_list)
        
        if use_detail_enhancement:
            # Detail enhancement modules for each level
            self.detail_modules = nn.ModuleList()
            for i, channels in enumerate(channels_list):
                # Stronger enhancement for shallow layers (high resolution)
                if i < self.num_levels // 2:
                    # Shallow layers: full detail enhancement
                    self.detail_modules.append(DetailEnhancementModule(channels))
                else:
                    # Deep layers: lighter detail enhancement (just multi-scale extraction)
                    self.detail_modules.append(MultiScaleDetailExtraction(channels))
        else:
            # Identity modules if disabled
            self.detail_modules = nn.ModuleList([nn.Identity() for _ in channels_list])
    
    def forward(self, features):
        """
        Args:
            features: List of features from each FPN level
        Returns:
            enhanced_features: List of detail-enhanced features
        """
        enhanced_features = []
        
        for i, feat in enumerate(features):
            enhanced = self.detail_modules[i](feat)
            enhanced_features.append(enhanced)
        
        return enhanced_features

