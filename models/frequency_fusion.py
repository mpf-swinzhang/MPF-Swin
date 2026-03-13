# --------------------------------------------------------
# Frequency Domain Fusion Module
# Uses Discrete Wavelet Transform (DWT) for multi-modal fusion
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DWT_2D(nn.Module):
    """2D Discrete Wavelet Transform using Haar wavelets."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input tensor
        Returns:
            ll: (B, C, H//2, W//2) low-frequency component
            lh: (B, C, H//2, W//2) horizontal high-frequency
            hl: (B, C, H//2, W//2) vertical high-frequency
            hh: (B, C, H//2, W//2) diagonal high-frequency
        """
        B, C, H, W = x.shape
        
        # Ensure even dimensions
        if H % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
            H += 1
        if W % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')
            W += 1
        
        # Reshape to process in 2x2 blocks
        # x: (B, C, H, W) -> (B, C, H//2, 2, W//2, 2)
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, C, H//2, W//2, 2, 2)
        x = x.view(B, C, H // 2, W // 2, 4)  # Flatten last two dims: [a, b, c, d] for each 2x2 block
        
        # Haar wavelet decomposition
        # For a 2x2 block [a, b; c, d]:
        # LL (low-low): (a+b+c+d)/2
        # LH (low-high): (a+b-c-d)/2  (horizontal detail)
        # HL (high-low): (a-b+c-d)/2  (vertical detail)
        # HH (high-high): (a-b-c+d)/2 (diagonal detail)
        
        # Extract elements: x[:, :, :, :, 0]=a, 1=b, 2=c, 3=d
        a = x[:, :, :, :, 0]
        b = x[:, :, :, :, 1]
        c = x[:, :, :, :, 2]
        d = x[:, :, :, :, 3]
        
        ll = (a + b + c + d) / 2.0  # Low-frequency
        lh = (a + b - c - d) / 2.0  # Horizontal high-frequency
        hl = (a - b + c - d) / 2.0  # Vertical high-frequency
        hh = (a - b - c + d) / 2.0  # Diagonal high-frequency
        
        return ll, lh, hl, hh


class IDWT_2D(nn.Module):
    """2D Inverse Discrete Wavelet Transform."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, ll, lh, hl, hh):
        """
        Args:
            ll: (B, C, H, W) low-frequency component
            lh: (B, C, H, W) horizontal high-frequency
            hl: (B, C, H, W) vertical high-frequency
            hh: (B, C, H, W) diagonal high-frequency
        Returns:
            x: (B, C, H*2, W*2) reconstructed tensor
        """
        B, C, H, W = ll.shape
        
        # Inverse Haar wavelet transform
        # Reconstruct 2x2 blocks from subbands
        # For subbands LL, LH, HL, HH, reconstruct [a, b; c, d]:
        # a = (LL + LH + HL + HH) / 2
        # b = (LL + LH - HL - HH) / 2
        # c = (LL - LH + HL - HH) / 2
        # d = (LL - LH - HL + HH) / 2
        
        a = (ll + lh + hl + hh) / 2.0
        b = (ll + lh - hl - hh) / 2.0
        c = (ll - lh + hl - hh) / 2.0
        d = (ll - lh - hl + hh) / 2.0
        
        # Stack to form 2x2 blocks
        x = torch.stack([a, b, c, d], dim=-1)  # (B, C, H, W, 4)
        
        # Reshape and interleave to reconstruct original size
        x = x.view(B, C, H, W, 2, 2)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, C, H, 2, W, 2)
        x = x.view(B, C, H * 2, W * 2)
        
        return x


class FrequencyDomainFusion(nn.Module):
    """Frequency domain fusion module for multi-modal features.
    
    Decomposes features into frequency subbands and fuses them separately:
    - Low-frequency (LL): Cross-modal attention for semantic fusion
    - High-frequency (LH/HL/HH): Max fusion for edge preservation
    """
    
    def __init__(self, channels, num_heads=8, dropout=0.1, use_high_freq=True):
        """
        Args:
            channels: Number of channels
            num_heads: Number of attention heads for low-frequency fusion
            dropout: Dropout rate
            use_high_freq: Whether to use high-frequency fusion
        """
        super().__init__()
        self.channels = channels
        self.use_high_freq = use_high_freq
        
        # DWT and IDWT
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        
        # Low-frequency fusion: Cross-modal attention
        self.low_freq_fusion = CrossModalAttentionLF(channels, num_heads, dropout)
        
        # High-frequency fusion: Max pooling + learnable weighting
        if use_high_freq:
            self.high_freq_weight = nn.Parameter(torch.ones(3) / 3.0)  # Weight for LH, HL, HH
            self.high_freq_conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),  # Depthwise
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1),  # Pointwise
                nn.BatchNorm2d(channels)
            )
        
        # Feature refinement after reconstruction
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
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
        
        # Decompose into frequency subbands
        opt_ll, opt_lh, opt_hl, opt_hh = self.dwt(optical_feat)
        sar_ll, sar_lh, sar_hl, sar_hh = self.dwt(sar_feat)
        
        # Low-frequency fusion: Cross-modal attention
        fused_ll = self.low_freq_fusion(opt_ll, sar_ll)
        
        # High-frequency fusion: Max pooling + weighted combination
        if self.use_high_freq:
            # Max fusion for high-frequency components
            fused_lh = torch.maximum(opt_lh, sar_lh)
            fused_hl = torch.maximum(opt_hl, sar_hl)
            fused_hh = torch.maximum(opt_hh, sar_hh)
            
            # Weighted combination
            weights = F.softmax(self.high_freq_weight, dim=0)
            fused_hf = (weights[0] * fused_lh + 
                       weights[1] * fused_hl + 
                       weights[2] * fused_hh)
            
            # Refine high-frequency
            fused_hf = self.high_freq_conv(fused_hf)
        else:
            # Simple max fusion if high-frequency disabled
            fused_lh = torch.maximum(opt_lh, sar_lh)
            fused_hl = torch.maximum(opt_hl, sar_hl)
            fused_hh = torch.maximum(opt_hh, sar_hh)
        
        # Reconstruct from frequency domain
        if self.use_high_freq:
            fused_feat = self.idwt(fused_ll, fused_lh, fused_hl, fused_hh)
        else:
            fused_feat = self.idwt(fused_ll, fused_lh, fused_hl, fused_hh)
        
        # Handle size mismatch (due to padding in DWT)
        if fused_feat.shape[2:] != (H, W):
            fused_feat = F.interpolate(fused_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # Refine fused features
        fused_feat = self.refine(fused_feat)
        
        return fused_feat


class CrossModalAttentionLF(nn.Module):
    """Cross-modal attention for low-frequency components."""
    
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
        
        # Lightweight FFN for low-frequency
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout)
        )
    
    def forward(self, optical_feat, sar_feat):
        """
        Args:
            optical_feat: (B, C, H, W) optical low-frequency features
            sar_feat: (B, C, H, W) SAR low-frequency features
        Returns:
            fused_feat: (B, C, H, W) fused low-frequency features
        """
        B, C, H, W = optical_feat.shape
        
        # Reshape to (B, H*W, C) for attention
        optical_flat = optical_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        sar_flat = sar_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Normalize
        optical_flat = self.norm1(optical_flat)
        sar_flat = self.norm1(sar_flat)
        
        # Cross-modal attention: Optical queries SAR
        q_opt = self.q_optical(optical_flat)
        k_sar = self.k_sar(sar_flat)
        v_sar = self.v_sar(sar_flat)
        
        # Reshape for multi-head attention
        q_opt = q_opt.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        k_sar = k_sar.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        v_sar = v_sar.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_opt = (q_opt @ k_sar.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_opt = F.softmax(attn_opt, dim=-1)
        attn_opt = self.dropout(attn_opt)
        
        out_opt = (attn_opt @ v_sar).transpose(1, 2).reshape(B, H*W, C)
        out_opt = self.proj(out_opt)
        out_opt = self.dropout(out_opt)
        out_opt = optical_flat + out_opt
        
        # Cross-modal attention: SAR queries Optical
        q_sar = self.q_sar(sar_flat)
        k_opt = self.k_optical(optical_flat)
        v_opt = self.v_optical(optical_flat)
        
        q_sar = q_sar.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        k_opt = k_opt.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        v_opt = v_opt.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_sar = (q_sar @ k_opt.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_sar = F.softmax(attn_sar, dim=-1)
        attn_sar = self.dropout(attn_sar)
        
        out_sar = (attn_sar @ v_opt).transpose(1, 2).reshape(B, H*W, C)
        out_sar = self.proj(out_sar)
        out_sar = self.dropout(out_sar)
        out_sar = sar_flat + out_sar
        
        # Combine both directions
        fused = out_opt + out_sar
        fused = self.norm2(fused)
        
        # FFN
        fused = fused + self.ffn(fused)
        
        # Reshape back to (B, C, H, W)
        fused = fused.transpose(1, 2).reshape(B, C, H, W)
        
        return fused


class ProgressiveFrequencyFusion(nn.Module):
    """Progressive frequency domain fusion for FPN decoder.
    
    Applies frequency domain fusion at each FPN level:
    - Shallow layers: Emphasize high-frequency (edges)
    - Deep layers: Emphasize low-frequency (semantics)
    """
    
    def __init__(self, channels_list, num_heads=8, dropout=0.1, use_frequency=True):
        """
        Args:
            channels_list: List of channel numbers for each FPN level
            Note: After lateral_conv, all features have the same channels (embed_dim)
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_frequency: Whether to enable frequency domain fusion
        """
        super().__init__()
        self.num_levels = len(channels_list)
        self.use_frequency = use_frequency
        
        if use_frequency:
            # Frequency domain fusion modules
            # Note: After lateral_conv, all features have embed_dim channels
            # So we use the first channel value (which should be embed_dim) for all levels
            actual_channels = channels_list[0] if len(channels_list) > 0 else 128
            self.freq_fusion_modules = nn.ModuleList()
            for i in range(self.num_levels):
                # All levels use the same channel number (embed_dim after lateral_conv)
                self.freq_fusion_modules.append(
                    FrequencyDomainFusion(actual_channels, num_heads, dropout, use_high_freq=True)
                )
        else:
            self.freq_fusion_modules = None
    
    def forward(self, optical_features, sar_features, spatial_fused_features=None):
        """
        Args:
            optical_features: List of optical features from each FPN level (after lateral_conv, embed_dim channels)
            sar_features: List of SAR features from each FPN level (after lateral_conv, embed_dim channels)
            spatial_fused_features: Optional pre-fused features from spatial domain (not used in current implementation)
        Returns:
            fused_features: List of frequency-domain fused features
        """
        if not self.use_frequency:
            return spatial_fused_features if spatial_fused_features is not None else optical_features
        
        freq_fused_features = []
        for i, (opt_feat, sar_feat) in enumerate(zip(optical_features, sar_features)):
            # Frequency domain fusion
            freq_fused = self.freq_fusion_modules[i](opt_feat, sar_feat)
            freq_fused_features.append(freq_fused)
        
        return freq_fused_features

