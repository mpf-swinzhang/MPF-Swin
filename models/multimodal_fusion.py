# --------------------------------------------------------
# Multi-modal Multi-scale Fusion Modules
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ElementWiseFusion(nn.Module):
    """Element-wise fusion for shallow layers (low-level features).
    
    Uses element-wise addition followed by 1x1 conv normalization.
    """
    
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, optical_feat, sar_feat):
        """
        Args:
            optical_feat: (B, C, H, W) optical features
            sar_feat: (B, C, H, W) SAR features
        Returns:
            fused_feat: (B, C, H, W) fused features
        """
        # Element-wise addition
        fused = optical_feat + sar_feat
        # Normalize and refine
        fused = self.norm(fused)
        fused = self.conv(fused)
        fused = self.relu(fused)
        return fused


class CrossModalAttention(nn.Module):
    """Cross-modal attention for deep layers (high-level semantic features).
    
    Query from one modality, Key/Value from another modality.
    """
    
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        # Query from optical, Key/Value from SAR
        self.q_optical = nn.Linear(channels, channels, bias=False)
        self.k_sar = nn.Linear(channels, channels, bias=False)
        self.v_sar = nn.Linear(channels, channels, bias=False)
        
        # Query from SAR, Key/Value from optical
        self.q_sar = nn.Linear(channels, channels, bias=False)
        self.k_optical = nn.Linear(channels, channels, bias=False)
        self.v_optical = nn.Linear(channels, channels, bias=False)
        
        # Output projection
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        
    def forward(self, optical_feat, sar_feat):
        """
        Args:
            optical_feat: (B, C, H, W) optical features
            sar_feat: (B, C, H, W) SAR features
        Returns:
            fused_feat: (B, C, H, W) fused features
        """
        B, C, H, W = optical_feat.shape
        
        # Reshape to (B, H*W, C) for attention
        optical_flat = optical_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        sar_flat = sar_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Normalize
        optical_flat = self.norm1(optical_flat)
        sar_flat = self.norm1(sar_flat)
        
        # Cross-modal attention: Optical queries SAR
        q_opt = self.q_optical(optical_flat)  # (B, H*W, C)
        k_sar = self.k_sar(sar_flat)  # (B, H*W, C)
        v_sar = self.v_sar(sar_flat)  # (B, H*W, C)
        
        # Reshape for multi-head attention
        q_opt = q_opt.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, H*W, head_dim)
        k_sar = k_sar.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        v_sar = v_sar.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_opt = (q_opt @ k_sar.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, H*W, H*W)
        attn_opt = F.softmax(attn_opt, dim=-1)
        attn_opt = self.dropout(attn_opt)
        
        out_opt = (attn_opt @ v_sar).transpose(1, 2).reshape(B, H*W, C)  # (B, H*W, C)
        out_opt = self.proj(out_opt)
        out_opt = self.dropout(out_opt)
        out_opt = optical_flat + out_opt  # Residual
        
        # Cross-modal attention: SAR queries Optical
        q_sar = self.q_sar(sar_flat)  # (B, H*W, C)
        k_opt = self.k_optical(optical_flat)  # (B, H*W, C)
        v_opt = self.v_optical(optical_flat)  # (B, H*W, C)
        
        # Reshape for multi-head attention
        q_sar = q_sar.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        k_opt = k_opt.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        v_opt = v_opt.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_sar = (q_sar @ k_opt.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_sar = F.softmax(attn_sar, dim=-1)
        attn_sar = self.dropout(attn_sar)
        
        out_sar = (attn_sar @ v_opt).transpose(1, 2).reshape(B, H*W, C)
        out_sar = self.proj(out_sar)
        out_sar = self.dropout(out_sar)
        out_sar = sar_flat + out_sar  # Residual
        
        # Combine both directions
        fused = out_opt + out_sar  # (B, H*W, C)
        fused = self.norm2(fused)
        
        # FFN
        fused = fused + self.ffn(fused)
        
        # Reshape back to (B, C, H, W)
        fused = fused.transpose(1, 2).reshape(B, C, H, W)
        
        return fused


class ProgressiveMultiScaleFusion(nn.Module):
    """Progressive multi-scale fusion module for FPN decoder.
    
    Uses different fusion strategies at different levels:
    - Shallow layers: Element-wise fusion
    - Deep layers: Cross-modal attention
    """
    
    def __init__(self, channels_list, num_heads=8, dropout=0.1):
        """
        Args:
            channels_list: List of channel numbers for each FPN level
            num_heads: Number of attention heads for deep layers
            dropout: Dropout rate
        """
        super().__init__()
        self.num_levels = len(channels_list)
        
        # Create fusion modules for each level
        self.fusion_modules = nn.ModuleList()
        for i, channels in enumerate(channels_list):
            if i < self.num_levels // 2:
                # Shallow layers: element-wise fusion
                self.fusion_modules.append(ElementWiseFusion(channels))
            else:
                # Deep layers: cross-modal attention
                self.fusion_modules.append(CrossModalAttention(channels, num_heads, dropout))
    
    def forward(self, optical_features, sar_features):
        """
        Args:
            optical_features: List of optical features from each FPN level
            sar_features: List of SAR features from each FPN level
        Returns:
            fused_features: List of fused features
        """
        assert len(optical_features) == len(sar_features) == self.num_levels
        
        fused_features = []
        for i, (opt_feat, sar_feat) in enumerate(zip(optical_features, sar_features)):
            fused = self.fusion_modules[i](opt_feat, sar_feat)
            fused_features.append(fused)
        
        return fused_features


