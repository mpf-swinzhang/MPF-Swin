# --------------------------------------------------------
# Multi-modal Multi-scale Segmentation Model
# Based on Swin Transformer with dual-branch encoder and progressive fusion
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .multimodal_fusion import ProgressiveMultiScaleFusion
from .frequency_fusion import ProgressiveFrequencyFusion
from .boundary_enhancement import ProgressiveBoundaryEnhancement
from .stage_fusion import ProgressiveMPFModule
from .detail_enhancement import ProgressiveDetailEnhancement


class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder with progressive multi-modal fusion."""
    
    def __init__(self, embed_dim, num_classes, img_size=224, fusion_modules=None, 
                 freq_fusion_modules=None, boundary_enhancement=None, 
                 use_detail_enhancement=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Feature dimensions at each stage: [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        # We'll work with 4 levels (stages 1-4)
        self.channels_list = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        
        # Spatial domain fusion modules (if provided)
        self.fusion_modules = fusion_modules
        
        # Frequency domain fusion modules (if provided)
        self.freq_fusion_modules = freq_fusion_modules
        
        # Boundary enhancement modules (if provided)
        self.boundary_enhancement = boundary_enhancement
        
        # Detail enhancement modules (NEW)
        self.use_detail_enhancement = use_detail_enhancement
        if use_detail_enhancement:
            self.detail_enhancement = ProgressiveDetailEnhancement(
                channels_list=[embed_dim] * 4,  # After lateral_conv, all have embed_dim
                use_detail_enhancement=True
            )
        else:
            self.detail_enhancement = None
        
        # Lateral connections (1x1 conv to reduce channels)
        # Will be created dynamically based on actual feature dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, embed_dim, kernel_size=1) for ch in self.channels_list
        ])
        
        # Store actual feature channels (will be set on first forward)
        self._actual_channels = None
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) for _ in range(len(self.channels_list) - 1)
        ])
        
        # Final classification layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1)
        )
    
    def forward(self, features, sar_features=None):
        """
        Args:
            features: List of features from optical branch [stage1, stage2, stage3, stage4]
            sar_features: List of features from SAR branch [stage1, stage2, stage3, stage4]
        Returns:
            Segmentation output: (B, num_classes, H, W)
        """
        # Apply lateral connections - check and fix channel mismatches
        laterals = []
        for i, (conv, feat) in enumerate(zip(self.lateral_convs, features)):
            actual_channels = feat.shape[1]
            expected_channels = self.channels_list[i]
            
            if actual_channels != expected_channels:
                # Channel mismatch - need to create a proper conv layer
                # Check if we already created a replacement conv
                if not hasattr(self, f'_lateral_conv_{i}') or getattr(self, f'_lateral_conv_{i}').in_channels != actual_channels:
                    # Create new conv and register it as a module so it's tracked by optimizer
                    new_conv = nn.Conv2d(actual_channels, self.embed_dim, kernel_size=1).to(feat.device)
                    # Initialize weights
                    nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                    if new_conv.bias is not None:
                        nn.init.constant_(new_conv.bias, 0)
                    # Ensure bias has same dtype as input (for AMP compatibility)
                    if new_conv.bias is not None:
                        new_conv.bias.data = new_conv.bias.data.to(dtype=feat.dtype)
                    self.add_module(f'_lateral_conv_{i}', new_conv)
                else:
                    # Layer exists, ensure dtype matches input
                    lateral_conv = getattr(self, f'_lateral_conv_{i}')
                    # Convert both weight and bias to match input dtype BEFORE calling forward
                    input_dtype = feat.dtype
                    if lateral_conv.weight.dtype != input_dtype:
                        lateral_conv.weight.data = lateral_conv.weight.data.to(dtype=input_dtype)
                    if lateral_conv.bias is not None:
                        if lateral_conv.bias.dtype != input_dtype:
                            lateral_conv.bias.data = lateral_conv.bias.data.to(dtype=input_dtype)
                laterals.append(getattr(self, f'_lateral_conv_{i}')(feat))
            else:
                laterals.append(conv(feat))
        
        # If SAR features provided, apply fusion
        if sar_features is not None:
            sar_laterals = []
            for i, feat in enumerate(sar_features):
                actual_channels = feat.shape[1]
                expected_channels = self.channels_list[i]
                
                # Check if we can reuse optical lateral conv
                optical_actual = features[i].shape[1]
                if actual_channels == optical_actual:
                    # Same channels as optical, can reuse the same conv
                    if actual_channels != expected_channels and hasattr(self, f'_lateral_conv_{i}'):
                        sar_laterals.append(getattr(self, f'_lateral_conv_{i}')(feat))
                    else:
                        sar_laterals.append(self.lateral_convs[i](feat))
                else:
                    # Different channels, need separate conv
                    if not hasattr(self, f'_sar_lateral_conv_{i}') or getattr(self, f'_sar_lateral_conv_{i}').in_channels != actual_channels:
                        new_conv = nn.Conv2d(actual_channels, self.embed_dim, kernel_size=1).to(feat.device)
                        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                        if new_conv.bias is not None:
                            nn.init.constant_(new_conv.bias, 0)
                        # Ensure bias has same dtype as input (for AMP compatibility)
                        if new_conv.bias is not None:
                            new_conv.bias.data = new_conv.bias.data.to(dtype=feat.dtype)
                        self.add_module(f'_sar_lateral_conv_{i}', new_conv)
                    else:
                        # Layer exists, ensure dtype matches input
                        lateral_conv = getattr(self, f'_sar_lateral_conv_{i}')
                        # Convert both weight and bias to match input dtype BEFORE calling forward
                        input_dtype = feat.dtype
                        if lateral_conv.weight.dtype != input_dtype:
                            lateral_conv.weight.data = lateral_conv.weight.data.to(dtype=input_dtype)
                        if lateral_conv.bias is not None:
                            if lateral_conv.bias.dtype != input_dtype:
                                lateral_conv.bias.data = lateral_conv.bias.data.to(dtype=input_dtype)
                    sar_laterals.append(getattr(self, f'_sar_lateral_conv_{i}')(feat))
            
            # Step 1: Boundary enhancement (applied to individual modalities before fusion)
            if self.boundary_enhancement is not None:
                # Enhance optical and SAR features separately
                optical_enhanced, optical_edges = self.boundary_enhancement(laterals)
                sar_enhanced, sar_edges = self.boundary_enhancement(sar_laterals)
                # Use enhanced features for fusion
                laterals = optical_enhanced
                sar_laterals = sar_enhanced
            
            # Step 2: Spatial domain fusion
            if self.fusion_modules is not None:
                spatial_fused = self.fusion_modules(laterals, sar_laterals)
            else:
                # Simple addition if no fusion modules
                spatial_fused = [opt + sar for opt, sar in zip(laterals, sar_laterals)]
            
            # Step 3: Frequency domain fusion (applied after spatial fusion)
            # Frequency fusion operates on original laterals (before spatial fusion) for better frequency representation
            if self.freq_fusion_modules is not None:
                # Apply frequency fusion to original laterals (optical and SAR separately)
                freq_fused = self.freq_fusion_modules(laterals, sar_laterals, spatial_fused_features=spatial_fused)
                # Combine spatial and frequency fusion (weighted combination)
                laterals = []
                for i in range(len(spatial_fused)):
                    # 85% spatial + 15% frequency (spatial is more stable)
                    combined = 0.85 * spatial_fused[i] + 0.15 * freq_fused[i]
                    laterals.append(combined)
            else:
                laterals = spatial_fused
        
        # Top-down pathway with upsampling
        # Start from the deepest level
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample deeper feature and add to current
            upsampled = self.upsample_layers[i](laterals[i + 1])
            # Ensure same spatial size
            if upsampled.shape[2:] != laterals[i].shape[2:]:
                upsampled = F.interpolate(upsampled, size=laterals[i].shape[2:], 
                                        mode='bilinear', align_corners=False)
            laterals[i] = laterals[i] + upsampled
        
        # Detail Enhancement (NEW): Enhance details at all levels
        if self.detail_enhancement is not None:
            laterals = self.detail_enhancement(laterals)
        
        # Use the shallowest (highest resolution) feature
        output = laterals[0]
        
        # Final classification
        output = self.final_conv(output)
        
        # Upsample to original image size
        output = F.interpolate(output, size=(self.img_size, self.img_size), 
                             mode='bilinear', align_corners=False)
        
        return output


class MultiModalSwinSeg(nn.Module):
    """Multi-modal multi-scale segmentation model with dual-branch Swin Transformer encoders."""
    
    def __init__(self, config, num_classes):
        super().__init__()
        
        # Get Swin Transformer config
        swin_config = config.MODEL.SWIN
        img_size = config.DATA.IMG_SIZE
        embed_dim = swin_config.EMBED_DIM
        
        # Optical branch: 10 channels
        self.optical_backbone = SwinTransformer(
            img_size=img_size,
            patch_size=swin_config.PATCH_SIZE,
            in_chans=10,  # 10-channel optical
            num_classes=0,
            embed_dim=embed_dim,
            depths=swin_config.DEPTHS,
            num_heads=swin_config.NUM_HEADS,
            window_size=swin_config.WINDOW_SIZE,
            mlp_ratio=swin_config.MLP_RATIO,
            qkv_bias=swin_config.QKV_BIAS,
            qk_scale=swin_config.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=swin_config.APE,
            patch_norm=swin_config.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            fused_window_process=config.FUSED_WINDOW_PROCESS
        )
        
        # SAR branch: 3 channels
        self.sar_backbone = SwinTransformer(
            img_size=img_size,
            patch_size=swin_config.PATCH_SIZE,
            in_chans=3,  # 3-channel SAR
            num_classes=0,
            embed_dim=embed_dim,
            depths=swin_config.DEPTHS,
            num_heads=swin_config.NUM_HEADS,
            window_size=swin_config.WINDOW_SIZE,
            mlp_ratio=swin_config.MLP_RATIO,
            qkv_bias=swin_config.QKV_BIAS,
            qk_scale=swin_config.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=swin_config.APE,
            patch_norm=swin_config.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            fused_window_process=config.FUSED_WINDOW_PROCESS
        )
        
        # Progressive multi-scale fusion (spatial domain)
        # Note: After lateral_conv in FPNDecoder, all features have embed_dim channels
        # So fusion modules should use embed_dim for all levels
        num_heads = getattr(config.MODEL, 'FUSION_NUM_HEADS', 8)
        dropout = getattr(config.MODEL, 'FUSION_DROPOUT', 0.1)
        num_levels = 4  # 4 stages
        fusion_channels_list = [embed_dim] * num_levels  # All levels use embed_dim after lateral_conv
        fusion_modules = ProgressiveMultiScaleFusion(
            channels_list=fusion_channels_list,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Frequency domain fusion
        # Note: After lateral_conv, all features have embed_dim channels
        # So frequency fusion modules should use embed_dim for all levels
        use_frequency = getattr(config.MODEL, 'USE_FREQUENCY_FUSION', True)
        freq_fusion_channels_list = [embed_dim] * num_levels  # All levels use embed_dim after lateral_conv
        freq_fusion_modules = ProgressiveFrequencyFusion(
            channels_list=freq_fusion_channels_list,
            num_heads=num_heads,
            dropout=dropout,
            use_frequency=use_frequency
        ) if use_frequency else None
        
        # Boundary enhancement
        use_boundary = getattr(config.MODEL, 'USE_BOUNDARY_ENHANCEMENT', True)
        boundary_channels_list = [embed_dim] * num_levels  # After lateral_conv, all have embed_dim
        boundary_enhancement = ProgressiveBoundaryEnhancement(
            channels_list=boundary_channels_list
        ) if use_boundary else None
        
        # Detail enhancement (NEW)
        use_detail = getattr(config.MODEL, 'USE_DETAIL_ENHANCEMENT', True)
        
        # Stage-level fusion (MPFModule) - fuse at each stage immediately
        use_stage_fusion = getattr(config.MODEL, 'USE_STAGE_FUSION', False)
        if use_stage_fusion:
            # Get channel dimensions for each stage
            # IMPORTANT: In forward_features_with_stage_fusion, we extract features AFTER each layer
            # Each layer includes Patch Merging at the end (except last stage)
            # So actual channels after layer() are: Stage1→2C, Stage2→4C, Stage3→8C, Stage4→8C
            channels_list_optical = [
                embed_dim * 2,  # Stage 1 (after Patch Merging)
                embed_dim * 4,  # Stage 2 (after Patch Merging)
                embed_dim * 8,  # Stage 3 (after Patch Merging)
                embed_dim * 8   # Stage 4 (no Patch Merging after last stage)
            ]
            channels_list_sar = channels_list_optical  # Same for SAR branch
            
            self.stage_fusion = ProgressiveMPFModule(
                channels_list_optical=channels_list_optical,
                channels_list_sar=channels_list_sar,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_boundary_enhancement=use_boundary,
                use_spatial_fusion=True,
                use_frequency_fusion=use_frequency,
                spatial_freq_weight=0.85
            )
            # For stage fusion, decoder doesn't need fusion modules (already fused)
            self.decoder = FPNDecoder(
                embed_dim=embed_dim,
                num_classes=num_classes,
                img_size=img_size,
                fusion_modules=None,  # No fusion needed, already fused per-stage
                freq_fusion_modules=None,
                boundary_enhancement=None,
                use_detail_enhancement=use_detail
            )
        else:
            # Original method: unified fusion after all stages
            self.stage_fusion = None
            self.decoder = FPNDecoder(
                embed_dim=embed_dim,
                num_classes=num_classes,
                img_size=img_size,
                fusion_modules=fusion_modules,
                freq_fusion_modules=freq_fusion_modules,
                boundary_enhancement=boundary_enhancement,
                use_detail_enhancement=use_detail
            )
        
        self.use_stage_fusion = use_stage_fusion
    
    def forward_features(self, backbone, x):
        """Extract features from backbone."""
        x = backbone.patch_embed(x)
        if backbone.ape:
            x = x + backbone.absolute_pos_embed
        x = backbone.pos_drop(x)
        
        # Get features from each stage
        features = []
        for i, layer in enumerate(backbone.layers):
            x = layer(x)
            # Reshape to (B, C, H, W) for each stage
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            if H * W != L:
                H = int(np.sqrt(L))
                W = L // H
            x_reshaped = x.transpose(1, 2).reshape(B, C, H, W)
            features.append(x_reshaped)
        
        return features
    
    def forward_features_with_stage_fusion(self, optical_backbone, sar_backbone, optical, sar):
        """Extract features and fuse at each stage using MPFModule."""
        # Patch embedding
        optical_x = optical_backbone.patch_embed(optical)
        sar_x = sar_backbone.patch_embed(sar)
        
        if optical_backbone.ape:
            optical_x = optical_x + optical_backbone.absolute_pos_embed
            sar_x = sar_x + sar_backbone.absolute_pos_embed
        
        optical_x = optical_backbone.pos_drop(optical_x)
        sar_x = sar_backbone.pos_drop(sar_x)
        
        # Extract and fuse at each stage
        fused_features = []
        
        for i, (optical_layer, sar_layer) in enumerate(zip(
            optical_backbone.layers, 
            sar_backbone.layers
        )):
            # Extract features from current stage
            optical_x = optical_layer(optical_x)
            sar_x = sar_layer(sar_x)
            
            # Reshape to (B, C, H, W)
            B, L, C_opt = optical_x.shape
            H = W = int(L ** 0.5)
            if H * W != L:
                H = int(np.sqrt(L))
                W = L // H
            
            optical_feat = optical_x.transpose(1, 2).reshape(B, C_opt, H, W)
            
            B, L, C_sar = sar_x.shape
            H = W = int(L ** 0.5)
            if H * W != L:
                H = int(np.sqrt(L))
                W = L // H
            
            sar_feat = sar_x.transpose(1, 2).reshape(B, C_sar, H, W)
            
            # Fuse at current stage (immediate interaction)
            fused = self.stage_fusion.mpf_modules[i](optical_feat, sar_feat)
            fused_features.append(fused)
        
        return fused_features
    
    def forward(self, optical, sar):
        """
        Args:
            optical: (B, 10, H, W) optical image
            sar: (B, 3, H, W) SAR image
        Returns:
            Segmentation output: (B, num_classes, H, W)
        """
        if self.use_stage_fusion:
            # Stage-level fusion: fuse at each stage immediately
            fused_features = self.forward_features_with_stage_fusion(
                self.optical_backbone,
                self.sar_backbone,
                optical,
                sar
            )
            # Decode fused features (no fusion needed in decoder)
            output = self.decoder(fused_features, sar_features=None)
        else:
            # Original method: extract all features first, then fuse
            optical_features = self.forward_features(self.optical_backbone, optical)
            sar_features = self.forward_features(self.sar_backbone, sar)
            
            # Decode with progressive fusion
            output = self.decoder(optical_features, sar_features)
        
        return output


class MultiModalSwinV2Seg(nn.Module):
    """Multi-modal multi-scale segmentation model with dual-branch Swin Transformer V2 encoders."""
    
    def __init__(self, config, num_classes):
        super().__init__()
        
        # Get Swin Transformer V2 config
        swinv2_config = config.MODEL.SWINV2
        img_size = config.DATA.IMG_SIZE
        embed_dim = swinv2_config.EMBED_DIM
        
        # Optical branch: 10 channels
        self.optical_backbone = SwinTransformerV2(
            img_size=img_size,
            patch_size=swinv2_config.PATCH_SIZE,
            in_chans=10,  # 10-channel optical
            num_classes=0,
            embed_dim=embed_dim,
            depths=swinv2_config.DEPTHS,
            num_heads=swinv2_config.NUM_HEADS,
            window_size=swinv2_config.WINDOW_SIZE,
            mlp_ratio=swinv2_config.MLP_RATIO,
            qkv_bias=swinv2_config.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=swinv2_config.APE,
            patch_norm=swinv2_config.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            pretrained_window_sizes=swinv2_config.PRETRAINED_WINDOW_SIZES
        )
        
        # SAR branch: 3 channels
        self.sar_backbone = SwinTransformerV2(
            img_size=img_size,
            patch_size=swinv2_config.PATCH_SIZE,
            in_chans=3,  # 3-channel SAR
            num_classes=0,
            embed_dim=embed_dim,
            depths=swinv2_config.DEPTHS,
            num_heads=swinv2_config.NUM_HEADS,
            window_size=swinv2_config.WINDOW_SIZE,
            mlp_ratio=swinv2_config.MLP_RATIO,
            qkv_bias=swinv2_config.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=swinv2_config.APE,
            patch_norm=swinv2_config.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            pretrained_window_sizes=swinv2_config.PRETRAINED_WINDOW_SIZES
        )
        
        # Progressive multi-scale fusion (spatial domain)
        # Note: After lateral_conv in FPNDecoder, all features have embed_dim channels
        # So fusion modules should use embed_dim for all levels
        num_heads = getattr(config.MODEL, 'FUSION_NUM_HEADS', 8)
        dropout = getattr(config.MODEL, 'FUSION_DROPOUT', 0.1)
        num_levels = 4  # 4 stages
        fusion_channels_list = [embed_dim] * num_levels  # All levels use embed_dim after lateral_conv
        fusion_modules = ProgressiveMultiScaleFusion(
            channels_list=fusion_channels_list,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Frequency domain fusion
        # Note: After lateral_conv, all features have embed_dim channels
        # So frequency fusion modules should use embed_dim for all levels
        use_frequency = getattr(config.MODEL, 'USE_FREQUENCY_FUSION', True)
        freq_fusion_channels_list = [embed_dim] * num_levels  # All levels use embed_dim after lateral_conv
        freq_fusion_modules = ProgressiveFrequencyFusion(
            channels_list=freq_fusion_channels_list,
            num_heads=num_heads,
            dropout=dropout,
            use_frequency=use_frequency
        ) if use_frequency else None
        
        # Boundary enhancement
        use_boundary = getattr(config.MODEL, 'USE_BOUNDARY_ENHANCEMENT', True)
        boundary_channels_list = [embed_dim] * num_levels  # After lateral_conv, all have embed_dim
        boundary_enhancement = ProgressiveBoundaryEnhancement(
            channels_list=boundary_channels_list
        ) if use_boundary else None
        
        # Detail enhancement (NEW)
        use_detail = getattr(config.MODEL, 'USE_DETAIL_ENHANCEMENT', True)
        
        # Stage-level fusion (MPFModule) - fuse at each stage immediately
        use_stage_fusion = getattr(config.MODEL, 'USE_STAGE_FUSION', False)
        if use_stage_fusion:
            # Get channel dimensions for each stage
            # IMPORTANT: In forward_features_with_stage_fusion, we extract features AFTER each layer
            # Each layer includes Patch Merging at the end (except last stage)
            # So actual channels after layer() are: Stage1→2C, Stage2→4C, Stage3→8C, Stage4→8C
            channels_list_optical = [
                embed_dim * 2,  # Stage 1 (after Patch Merging)
                embed_dim * 4,  # Stage 2 (after Patch Merging)
                embed_dim * 8,  # Stage 3 (after Patch Merging)
                embed_dim * 8   # Stage 4 (no Patch Merging after last stage)
            ]
            channels_list_sar = channels_list_optical  # Same for SAR branch
            
            self.stage_fusion = ProgressiveMPFModule(
                channels_list_optical=channels_list_optical,
                channels_list_sar=channels_list_sar,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_boundary_enhancement=use_boundary,
                use_spatial_fusion=True,
                use_frequency_fusion=use_frequency,
                spatial_freq_weight=0.85
            )
            # For stage fusion, decoder doesn't need fusion modules (already fused)
            self.decoder = FPNDecoder(
                embed_dim=embed_dim,
                num_classes=num_classes,
                img_size=img_size,
                fusion_modules=None,  # No fusion needed, already fused per-stage
                freq_fusion_modules=None,
                boundary_enhancement=None,
                use_detail_enhancement=use_detail
            )
        else:
            # Original method: unified fusion after all stages
            self.stage_fusion = None
            self.decoder = FPNDecoder(
                embed_dim=embed_dim,
                num_classes=num_classes,
                img_size=img_size,
                fusion_modules=fusion_modules,
                freq_fusion_modules=freq_fusion_modules,
                boundary_enhancement=boundary_enhancement,
                use_detail_enhancement=use_detail
            )
        
        self.use_stage_fusion = use_stage_fusion
    
    def forward_features(self, backbone, x):
        """Extract features from backbone."""
        x = backbone.patch_embed(x)
        if backbone.ape:
            x = x + backbone.absolute_pos_embed
        x = backbone.pos_drop(x)
        
        # Get features from each stage
        features = []
        for i, layer in enumerate(backbone.layers):
            x = layer(x)
            # Reshape to (B, C, H, W) for each stage
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            if H * W != L:
                H = int(np.sqrt(L))
                W = L // H
            x_reshaped = x.transpose(1, 2).reshape(B, C, H, W)
            features.append(x_reshaped)
        
        return features
    
    def forward_features_with_stage_fusion(self, optical_backbone, sar_backbone, optical, sar):
        """Extract features and fuse at each stage using MPFModule."""
        # Patch embedding
        optical_x = optical_backbone.patch_embed(optical)
        sar_x = sar_backbone.patch_embed(sar)
        
        if optical_backbone.ape:
            optical_x = optical_x + optical_backbone.absolute_pos_embed
            sar_x = sar_x + sar_backbone.absolute_pos_embed
        
        optical_x = optical_backbone.pos_drop(optical_x)
        sar_x = sar_backbone.pos_drop(sar_x)
        
        # Extract and fuse at each stage
        fused_features = []
        
        for i, (optical_layer, sar_layer) in enumerate(zip(
            optical_backbone.layers, 
            sar_backbone.layers
        )):
            # Extract features from current stage
            optical_x = optical_layer(optical_x)
            sar_x = sar_layer(sar_x)
            
            # Reshape to (B, C, H, W)
            B, L, C_opt = optical_x.shape
            H = W = int(L ** 0.5)
            if H * W != L:
                H = int(np.sqrt(L))
                W = L // H
            
            optical_feat = optical_x.transpose(1, 2).reshape(B, C_opt, H, W)
            
            B, L, C_sar = sar_x.shape
            H = W = int(L ** 0.5)
            if H * W != L:
                H = int(np.sqrt(L))
                W = L // H
            
            sar_feat = sar_x.transpose(1, 2).reshape(B, C_sar, H, W)
            
            # Fuse at current stage (immediate interaction)
            fused = self.stage_fusion.mpf_modules[i](optical_feat, sar_feat)
            fused_features.append(fused)
        
        return fused_features
    
    def forward(self, optical, sar):
        """
        Args:
            optical: (B, 10, H, W) optical image
            sar: (B, 3, H, W) SAR image
        Returns:
            Segmentation output: (B, num_classes, H, W)
        """
        if self.use_stage_fusion:
            # Stage-level fusion: fuse at each stage immediately
            fused_features = self.forward_features_with_stage_fusion(
                self.optical_backbone,
                self.sar_backbone,
                optical,
                sar
            )
            # Decode fused features (no fusion needed in decoder)
            output = self.decoder(fused_features, sar_features=None)
        else:
            # Original method: extract all features first, then fuse
            optical_features = self.forward_features(self.optical_backbone, optical)
            sar_features = self.forward_features(self.sar_backbone, sar)
            
            # Decode with progressive fusion
            output = self.decoder(optical_features, sar_features)
        
        return output

