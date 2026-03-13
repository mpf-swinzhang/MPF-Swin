# --------------------------------------------------------
# Boundary Enhancement Module
# For improving edge detection and boundary accuracy in segmentation
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedChannelTransformation(nn.Module):
    """Gated Channel Transformation for edge feature extraction."""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Gate mechanism
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Edge extraction
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),  # Depthwise
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input features
        Returns:
            edge_feat: (B, C, H, W) edge features
        """
        # Gate mechanism to focus on edge regions
        gate = self.gate_conv(x)
        
        # Extract edge features
        edge = self.edge_conv(x)
        
        # Apply gate
        edge_feat = gate * edge + x  # Residual connection
        
        return edge_feat


class CoordinateAttention(nn.Module):
    """Coordinate Attention for spatial and channel attention."""
    
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Pooling layers
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Shared MLP
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        # Separate convs for h and w
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input features
        Returns:
            out: (B, C, H, W) attended features
        """
        identity = x
        
        n, c, h, w = x.size()
        
        # Pool along height and width
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, 1, W) -> (B, C, W, 1)
        
        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        
        # Shared MLP
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split and process
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, 1, W)
        
        # Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (B, C, 1, W)
        
        # Apply attention
        out = identity * a_h * a_w
        
        return out


class BoundaryAwarenessModule(nn.Module):
    """Boundary Awareness Module for shallow layers (low-level features).
    
    Uses Gated Channel Transformation to extract edge features from low-level features.
    """
    
    def __init__(self, channels):
        super().__init__()
        self.gct = GatedChannelTransformation(channels)
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, feat):
        """
        Args:
            feat: (B, C, H, W) low-level features
        Returns:
            edge_feat: (B, C, H, W) edge-aware features
        """
        edge_feat = self.gct(feat)
        edge_feat = self.norm(edge_feat)
        edge_feat = self.relu(edge_feat)
        return edge_feat


class BoundaryGuidedFusion(nn.Module):
    """Boundary-Guided Fusion for deep layers.
    
    Uses edge features to guide semantic feature fusion.
    """
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Depthwise separable convolution for efficiency
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Coordinate attention
        self.coord_attn = CoordinateAttention(channels)
        
        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, semantic_feat, edge_feat):
        """
        Args:
            semantic_feat: (B, C, H, W) semantic features
            edge_feat: (B, C, H, W) edge features
        Returns:
            fused_feat: (B, C, H, W) boundary-guided fused features
        """
        # Concatenate semantic and edge features
        concat_feat = semantic_feat + edge_feat  # Element-wise addition
        
        # Depthwise separable convolution
        fused = self.depthwise(concat_feat)
        fused = self.bn1(fused)
        fused = self.pointwise(fused)
        
        # Coordinate attention
        fused = self.coord_attn(fused)
        
        # Refinement
        fused = self.refine(fused)
        
        # Residual connection
        fused = fused + semantic_feat
        
        return fused


class ProgressiveBoundaryEnhancement(nn.Module):
    """Progressive Boundary Enhancement for FPN decoder.
    
    Applies boundary enhancement at different levels:
    - Shallow layers: Boundary Awareness Module (edge extraction)
    - Deep layers: Boundary-Guided Fusion (edge-guided semantic fusion)
    """
    
    def __init__(self, channels_list):
        """
        Args:
            channels_list: List of channel numbers for each FPN level
        """
        super().__init__()
        self.num_levels = len(channels_list)
        
        # Boundary awareness modules for shallow layers (edge extraction)
        self.boundary_aware_modules = nn.ModuleList()
        for i, channels in enumerate(channels_list):
            if i < self.num_levels // 2:  # Shallow layers
                self.boundary_aware_modules.append(BoundaryAwarenessModule(channels))
            else:  # Deep layers: placeholder (will use boundary-guided fusion)
                self.boundary_aware_modules.append(nn.Identity())
        
        # Boundary-guided fusion modules for deep layers
        self.boundary_guided_fusions = nn.ModuleList()
        for i, channels in enumerate(channels_list):
            if i >= self.num_levels // 2:  # Deep layers
                self.boundary_guided_fusions.append(BoundaryGuidedFusion(channels))
            else:  # Shallow layers: placeholder
                self.boundary_guided_fusions.append(nn.Identity())
    
    def forward(self, features):
        """
        Args:
            features: List of features from each FPN level
        Returns:
            enhanced_features: List of boundary-enhanced features
        """
        enhanced_features = []
        edge_features = []
        
        # First pass: extract edge features from all levels
        for i, feat in enumerate(features):
            edge_feat = self.boundary_aware_modules[i](feat)
            edge_features.append(edge_feat)
        
        # Second pass: apply boundary-guided fusion for deep layers
        for i, (feat, edge_feat) in enumerate(zip(features, edge_features)):
            if i >= self.num_levels // 2:  # Deep layers
                # Use boundary-guided fusion
                enhanced = self.boundary_guided_fusions[i](feat, edge_feat)
            else:  # Shallow layers
                # Just use edge-enhanced features
                enhanced = edge_feat
            enhanced_features.append(enhanced)
        
        return enhanced_features, edge_features


