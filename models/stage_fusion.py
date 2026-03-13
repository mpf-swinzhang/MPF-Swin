# --------------------------------------------------------
# MPF Module (Multi-modal Pyramid Fusion)
# Combines FPN lateral, boundary enhancement, spatial fusion, and frequency fusion
# for per-stage feature interaction
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_fusion import ElementWiseFusion, CrossModalAttention
from .frequency_fusion import FrequencyDomainFusion
from .boundary_enhancement import BoundaryAwarenessModule, BoundaryGuidedFusion


class MPFModule(nn.Module):
    """MPF Module (Multi-modal Pyramid Fusion)
    
    Stage-level fusion module that combines:
    1. FPN lateral connection (channel alignment)
    2. Boundary enhancement
    3. Spatial fusion (element-wise or cross-modal attention)
    4. Frequency fusion
    
    This module is applied at each stage after feature extraction.
    """
    
    def __init__(self, 
                 in_channels_optical, 
                 in_channels_sar, 
                 out_channels,
                 stage_level=0,  # 0=shallow, 3=deep
                 num_heads=8,
                 dropout=0.1,
                 use_boundary_enhancement=True,
                 use_spatial_fusion=True,
                 use_frequency_fusion=True,
                 spatial_freq_weight=0.85):
        """
        Args:
            in_channels_optical: Input channels for optical branch
            in_channels_sar: Input channels for SAR branch
            out_channels: Output channels (embed_dim for FPN)
            stage_level: Stage level (0=shallow, 3=deep) to determine fusion strategy
            num_heads: Number of attention heads for deep layers
            dropout: Dropout rate
            use_boundary_enhancement: Whether to use boundary enhancement
            use_spatial_fusion: Whether to use spatial fusion
            use_frequency_fusion: Whether to use frequency fusion
            spatial_freq_weight: Weight for spatial fusion (1 - weight for frequency)
        """
        super().__init__()
        self.stage_level = stage_level
        self.out_channels = out_channels
        self.use_boundary_enhancement = use_boundary_enhancement
        self.use_spatial_fusion = use_spatial_fusion
        self.use_frequency_fusion = use_frequency_fusion
        self.spatial_freq_weight = spatial_freq_weight
        
        # Step 1: FPN Lateral Connections (channel alignment)
        # Optical branch lateral
        if in_channels_optical != out_channels:
            self.optical_lateral = nn.Conv2d(in_channels_optical, out_channels, kernel_size=1)
            nn.init.kaiming_normal_(self.optical_lateral.weight, mode='fan_out', nonlinearity='relu')
            if self.optical_lateral.bias is not None:
                nn.init.constant_(self.optical_lateral.bias, 0)
        else:
            self.optical_lateral = nn.Identity()
        
        # SAR branch lateral
        if in_channels_sar != out_channels:
            self.sar_lateral = nn.Conv2d(in_channels_sar, out_channels, kernel_size=1)
            nn.init.kaiming_normal_(self.sar_lateral.weight, mode='fan_out', nonlinearity='relu')
            if self.sar_lateral.bias is not None:
                nn.init.constant_(self.sar_lateral.bias, 0)
        else:
            self.sar_lateral = nn.Identity()
        
        # Step 2: Boundary Enhancement
        if use_boundary_enhancement:
            if stage_level < 2:  # Shallow layers: Boundary Awareness
                self.boundary_enhancement_optical = BoundaryAwarenessModule(out_channels)
                self.boundary_enhancement_sar = BoundaryAwarenessModule(out_channels)
            else:  # Deep layers: Boundary-Guided Fusion (needs edge extractor)
                self.boundary_enhancement_optical = BoundaryGuidedFusion(out_channels)
                self.boundary_enhancement_sar = BoundaryGuidedFusion(out_channels)
                # Edge extractor for deep layers (to extract edge features before fusion)
                self._edge_extractor = BoundaryAwarenessModule(out_channels)
        else:
            self.boundary_enhancement_optical = nn.Identity()
            self.boundary_enhancement_sar = nn.Identity()
        
        # Step 3: Spatial Fusion
        if use_spatial_fusion:
            if stage_level < 2:  # Shallow layers: Element-wise fusion
                self.spatial_fusion = ElementWiseFusion(out_channels)
            else:  # Deep layers: Cross-modal attention
                self.spatial_fusion = CrossModalAttention(out_channels, num_heads, dropout)
        else:
            self.spatial_fusion = None
        
        # Step 4: Frequency Fusion
        if use_frequency_fusion:
            self.frequency_fusion = FrequencyDomainFusion(
                out_channels, 
                num_heads=num_heads, 
                dropout=dropout,
                use_high_freq=True
            )
        else:
            self.frequency_fusion = None
        
        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, optical_feat, sar_feat):
        """
        Args:
            optical_feat: (B, in_channels_optical, H, W) optical features from current stage
            sar_feat: (B, in_channels_sar, H, W) SAR features from current stage
        Returns:
            fused_feat: (B, out_channels, H, W) fused features
        """
        # Step 1: FPN Lateral Connections (channel alignment)
        # Check if lateral connections need to be created dynamically
        # This handles cases where actual input channels don't match expected channels
        actual_opt_channels = optical_feat.shape[1]
        actual_sar_channels = sar_feat.shape[1]
        
        # Optical lateral
        if actual_opt_channels != self.out_channels:
            # Check if we have a pre-created lateral for this channel size
            lateral_key_opt = f'_lateral_opt_{actual_opt_channels}'
            if hasattr(self, lateral_key_opt):
                lateral_conv = getattr(self, lateral_key_opt)
                # Ensure entire layer dtype matches input (important for AMP)
                # Convert both weight and bias to match input dtype BEFORE calling forward
                input_dtype = optical_feat.dtype
                if lateral_conv.weight.dtype != input_dtype:
                    lateral_conv.weight.data = lateral_conv.weight.data.to(dtype=input_dtype)
                if lateral_conv.bias is not None:
                    if lateral_conv.bias.dtype != input_dtype:
                        lateral_conv.bias.data = lateral_conv.bias.data.to(dtype=input_dtype)
                # Double check before forward
                assert lateral_conv.weight.dtype == input_dtype, f"Weight dtype mismatch: {lateral_conv.weight.dtype} vs {input_dtype}"
                if lateral_conv.bias is not None:
                    assert lateral_conv.bias.dtype == input_dtype, f"Bias dtype mismatch: {lateral_conv.bias.dtype} vs {input_dtype}"
                optical_lateral = lateral_conv(optical_feat)
            else:
                # Create lateral conv dynamically and register it
                # This should only happen if channels don't match expected
                lateral_conv = nn.Conv2d(actual_opt_channels, self.out_channels, kernel_size=1)
                nn.init.kaiming_normal_(lateral_conv.weight, mode='fan_out', nonlinearity='relu')
                if lateral_conv.bias is not None:
                    nn.init.constant_(lateral_conv.bias, 0)
                # Move to device and ensure dtype consistency (important for AMP)
                lateral_conv = lateral_conv.to(optical_feat.device)
                # Ensure bias has same dtype as input (for AMP compatibility)
                if lateral_conv.bias is not None:
                    lateral_conv.bias.data = lateral_conv.bias.data.to(dtype=optical_feat.dtype)
                self.add_module(lateral_key_opt, lateral_conv)
                optical_lateral = lateral_conv(optical_feat)
        else:
            optical_lateral = self.optical_lateral(optical_feat)
        
        # SAR lateral
        if actual_sar_channels != self.out_channels:
            lateral_key_sar = f'_lateral_sar_{actual_sar_channels}'
            if hasattr(self, lateral_key_sar):
                lateral_conv = getattr(self, lateral_key_sar)
                # Ensure entire layer dtype matches input (important for AMP)
                # Convert both weight and bias to match input dtype BEFORE calling forward
                input_dtype = sar_feat.dtype
                if lateral_conv.weight.dtype != input_dtype:
                    lateral_conv.weight.data = lateral_conv.weight.data.to(dtype=input_dtype)
                if lateral_conv.bias is not None:
                    if lateral_conv.bias.dtype != input_dtype:
                        lateral_conv.bias.data = lateral_conv.bias.data.to(dtype=input_dtype)
                # Double check before forward
                assert lateral_conv.weight.dtype == input_dtype, f"Weight dtype mismatch: {lateral_conv.weight.dtype} vs {input_dtype}"
                if lateral_conv.bias is not None:
                    assert lateral_conv.bias.dtype == input_dtype, f"Bias dtype mismatch: {lateral_conv.bias.dtype} vs {input_dtype}"
                sar_lateral = lateral_conv(sar_feat)
            else:
                # Create lateral conv dynamically and register it
                lateral_conv = nn.Conv2d(actual_sar_channels, self.out_channels, kernel_size=1)
                nn.init.kaiming_normal_(lateral_conv.weight, mode='fan_out', nonlinearity='relu')
                if lateral_conv.bias is not None:
                    nn.init.constant_(lateral_conv.bias, 0)
                # Move to device and ensure dtype consistency (important for AMP)
                lateral_conv = lateral_conv.to(sar_feat.device)
                # Ensure bias has same dtype as input (for AMP compatibility)
                if lateral_conv.bias is not None:
                    lateral_conv.bias.data = lateral_conv.bias.data.to(dtype=sar_feat.dtype)
                self.add_module(lateral_key_sar, lateral_conv)
                sar_lateral = lateral_conv(sar_feat)
        else:
            sar_lateral = self.sar_lateral(sar_feat)
        
        # Ensure output channels are correct
        assert optical_lateral.shape[1] == self.out_channels, \
            f"Optical lateral output channels mismatch: expected {self.out_channels}, got {optical_lateral.shape[1]}. " \
            f"Input: {actual_opt_channels}, Expected input: {self.optical_lateral.in_channels if hasattr(self.optical_lateral, 'in_channels') else 'N/A'}"
        assert sar_lateral.shape[1] == self.out_channels, \
            f"SAR lateral output channels mismatch: expected {self.out_channels}, got {sar_lateral.shape[1]}. " \
            f"Input: {actual_sar_channels}, Expected input: {self.sar_lateral.in_channels if hasattr(self.sar_lateral, 'in_channels') else 'N/A'}"
        
        # Step 2: Boundary Enhancement (applied separately to each modality)
        if self.use_boundary_enhancement:
            if self.stage_level < 2:
                # Shallow layers: Boundary Awareness (edge extraction only)
                optical_enhanced = self.boundary_enhancement_optical(optical_lateral)
                sar_enhanced = self.boundary_enhancement_sar(sar_lateral)
            else:
                # Deep layers: Boundary-Guided Fusion (needs edge features first)
                # Extract edge features using pre-created edge extractor
                optical_edge = self._edge_extractor(optical_lateral)
                sar_edge = self._edge_extractor(sar_lateral)
                # Then apply boundary-guided fusion
                optical_enhanced = self.boundary_enhancement_optical(optical_lateral, optical_edge)
                sar_enhanced = self.boundary_enhancement_sar(sar_lateral, sar_edge)
        else:
            optical_enhanced = optical_lateral
            sar_enhanced = sar_lateral
        
        # Step 3: Spatial Fusion
        if self.use_spatial_fusion and self.spatial_fusion is not None:
            spatial_fused = self.spatial_fusion(optical_enhanced, sar_enhanced)
        else:
            # Simple addition if spatial fusion disabled
            spatial_fused = optical_enhanced + sar_enhanced
        
        # Step 4: Frequency Fusion
        if self.use_frequency_fusion and self.frequency_fusion is not None:
            # Frequency fusion operates on boundary-enhanced features
            freq_fused = self.frequency_fusion(optical_enhanced, sar_enhanced)
            
            # Combine spatial and frequency fusion
            fused = (self.spatial_freq_weight * spatial_fused + 
                    (1 - self.spatial_freq_weight) * freq_fused)
        else:
            fused = spatial_fused
        
        # Final refinement
        fused = self.refine(fused)
        
        return fused


class ProgressiveMPFModule(nn.Module):
    """Progressive MPF Module for all 4 stages.
    
    Creates an MPFModule for each stage with appropriate fusion strategies.
    """
    
    def __init__(self,
                 channels_list_optical,  # [C1, C2, C3, C4] for optical branch
                 channels_list_sar,      # [C1, C2, C3, C4] for SAR branch
                 embed_dim,              # Output channels (FPN embed_dim)
                 num_heads=8,
                 dropout=0.1,
                 use_boundary_enhancement=True,
                 use_spatial_fusion=True,
                 use_frequency_fusion=True,
                 spatial_freq_weight=0.85):
        """
        Args:
            channels_list_optical: List of channel numbers for each stage in optical branch
            channels_list_sar: List of channel numbers for each stage in SAR branch
            embed_dim: Output channels (FPN embed_dim)
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_boundary_enhancement: Whether to use boundary enhancement
            use_spatial_fusion: Whether to use spatial fusion
            use_frequency_fusion: Whether to use frequency fusion
            spatial_freq_weight: Weight for spatial fusion
        """
        super().__init__()
        self.num_stages = len(channels_list_optical)
        assert len(channels_list_optical) == len(channels_list_sar), \
            "Optical and SAR channel lists must have same length"
        
        # Create fusion module for each stage
        self.mpf_modules = nn.ModuleList()
        for i in range(self.num_stages):
            mpf_module = MPFModule(
                in_channels_optical=channels_list_optical[i],
                in_channels_sar=channels_list_sar[i],
                out_channels=embed_dim,
                stage_level=i,
                num_heads=num_heads,
                dropout=dropout,
                use_boundary_enhancement=use_boundary_enhancement,
                use_spatial_fusion=use_spatial_fusion,
                use_frequency_fusion=use_frequency_fusion,
                spatial_freq_weight=spatial_freq_weight
            )
            self.mpf_modules.append(mpf_module)
        
        # Pre-create dynamic lateral connections for all possible channel combinations
        # This avoids runtime creation issues with AMP
        self._precreate_dynamic_laterals(channels_list_optical, channels_list_sar, embed_dim)
    
    def _precreate_dynamic_laterals(self, channels_list_optical, channels_list_sar, embed_dim):
        """Pre-create dynamic lateral connections to avoid AMP issues."""
        unique_opt_channels = set(channels_list_optical)
        unique_sar_channels = set(channels_list_sar)
        
        for ch in unique_opt_channels:
            if ch != embed_dim:
                lateral_key = f'_lateral_opt_{ch}'
                if not hasattr(self, lateral_key):
                    lateral_conv = nn.Conv2d(ch, embed_dim, kernel_size=1)
                    nn.init.kaiming_normal_(lateral_conv.weight, mode='fan_out', nonlinearity='relu')
                    if lateral_conv.bias is not None:
                        nn.init.constant_(lateral_conv.bias, 0)
                    self.add_module(lateral_key, lateral_conv)
        
        for ch in unique_sar_channels:
            if ch != embed_dim:
                lateral_key = f'_lateral_sar_{ch}'
                if not hasattr(self, lateral_key):
                    lateral_conv = nn.Conv2d(ch, embed_dim, kernel_size=1)
                    nn.init.kaiming_normal_(lateral_conv.weight, mode='fan_out', nonlinearity='relu')
                    if lateral_conv.bias is not None:
                        nn.init.constant_(lateral_conv.bias, 0)
                    self.add_module(lateral_key, lateral_conv)
    
    def forward(self, optical_features, sar_features):
        """
        Args:
            optical_features: List of optical features from each stage
            sar_features: List of SAR features from each stage
        Returns:
            fused_features: List of fused features from each stage
        """
        assert len(optical_features) == len(sar_features) == self.num_stages, \
            f"Expected {self.num_stages} features, got {len(optical_features)} optical and {len(sar_features)} SAR"
        
        fused_features = []
        for i, (opt_feat, sar_feat) in enumerate(zip(optical_features, sar_features)):
            fused = self.mpf_modules[i](opt_feat, sar_feat)
            fused_features.append(fused)
        
        return fused_features

