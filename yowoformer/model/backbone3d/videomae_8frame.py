"""
VideoMAE Backbone for 8 Frames Input
Supports true 8-frame input by interpolating positional embeddings

Key Changes from 16-frame version:
1. Interpolate positional embeddings from 1568 -> 784 tokens
2. Reduce GFLOPs by ~50%
3. Increase FPS by ~2x

Architecture:
- Input: (B, 3, 8, 224, 224)
- Tubelet: 8 frames / 2 = 4 temporal tokens
- Spatial: 14 x 14 = 196 tokens
- Total: 4 x 196 = 784 tokens (vs 1568 for 16 frames)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel
import math
from typing import Optional, Tuple, Dict
from einops import rearrange, repeat


class VideoMAE8FrameBackbone(nn.Module):
    """
    VideoMAE backbone optimized for 8-frame input
    Uses positional embedding interpolation to support 8 frames
    while leveraging pretrained 16-frame weights
    """

    def __init__(self,
                 model_variant: str = "base",
                 pretrained: bool = True,
                 freeze_encoder: bool = True,
                 target_channels: int = 512,
                 target_spatial_size: int = 7,
                 dropout_rate: float = 0.2,
                 num_frames: int = 8):
        super().__init__()

        # Model selection
        self.model_map = {
            "base": "MCG-NJU/videomae-base-finetuned-kinetics",
            "base_raw": "MCG-NJU/videomae-base",
            "large": "MCG-NJU/videomae-large-finetuned-kinetics",
        }

        self.model_name = self.model_map.get(model_variant, model_variant)
        self.target_channels = target_channels
        self.target_spatial_size = target_spatial_size
        self.num_frames = num_frames
        self.pretrained_frames = 16  # VideoMAE pretrained uses 16 frames

        # VideoMAE uses tubelet_size=2, so temporal tokens = frames / 2
        self.temporal_tokens_pretrained = self.pretrained_frames // 2  # 8
        self.temporal_tokens_target = self.num_frames // 2  # 4
        self.spatial_tokens = 14 * 14  # 196

        print(f"[VideoMAE-8Frame] Initializing VideoMAE backbone")
        print(f"[VideoMAE-8Frame] Model: {self.model_name}")
        print(f"[VideoMAE-8Frame] Input frames: {num_frames}")
        print(f"[VideoMAE-8Frame] Temporal tokens: {self.temporal_tokens_target} (from {self.temporal_tokens_pretrained})")
        print(f"[VideoMAE-8Frame] Total tokens: {self.temporal_tokens_target * self.spatial_tokens} (from {self.temporal_tokens_pretrained * self.spatial_tokens})")
        print(f"[VideoMAE-8Frame] Target output: ({target_channels}, 1, {target_spatial_size}, {target_spatial_size})")

        # Load VideoMAE model
        try:
            self.videomae = VideoMAEModel.from_pretrained(self.model_name)
            self.hidden_size = self.videomae.config.hidden_size  # 768 for base
            print(f"[VideoMAE-8Frame] Model loaded successfully. Hidden size: {self.hidden_size}")
        except Exception as e:
            print(f"[VideoMAE-8Frame] Error loading model: {e}")
            raise

        # Interpolate positional embeddings for 8 frames
        if num_frames != self.pretrained_frames:
            self._interpolate_pos_embed()

        # Freeze encoder to save memory
        if freeze_encoder:
            for param in self.videomae.parameters():
                param.requires_grad = False
            print(f"[VideoMAE-8Frame] Encoder frozen")

        # Projection layers (same as original)
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size * 2, target_channels * target_spatial_size * target_spatial_size),
            nn.LayerNorm(target_channels * target_spatial_size * target_spatial_size)
        )

        # Spatial refinement
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(target_channels, target_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_channels, target_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_channels, target_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(target_channels)
        )

        self._init_weights()
        print(f"[VideoMAE-8Frame] Adapter layers initialized")

    def _interpolate_pos_embed(self):
        """
        Interpolate positional embeddings from 16 frames to 8 frames

        VideoMAE has NO CLS token in positional embeddings!
        Original: (1, 8*14*14, 768) = (1, 1568, 768) for 16 frames
        Target:   (1, 4*14*14, 768) = (1, 784, 768) for 8 frames
        """
        print(f"[VideoMAE-8Frame] Interpolating positional embeddings...")

        # Get original positional embeddings
        pos_embed = self.videomae.embeddings.position_embeddings  # (1, 1568, 768)
        print(f"[VideoMAE-8Frame]   Original pos_embed shape: {pos_embed.shape}")

        # VideoMAE does NOT have CLS token - all tokens are patch tokens
        # Shape: (1, T*H*W, C) = (1, 8*14*14, 768) = (1, 1568, 768)
        patch_pos = pos_embed  # (1, 1568, 768)

        # Reshape to (1, T, H, W, C)
        T_orig = self.temporal_tokens_pretrained  # 8
        H = W = 14
        C = self.hidden_size

        patch_pos = patch_pos.reshape(1, T_orig, H, W, C)

        # Interpolate temporal dimension: 8 -> 4
        T_new = self.temporal_tokens_target  # 4

        # Permute to (1, C, T, H, W) for interpolation
        patch_pos = patch_pos.permute(0, 4, 1, 2, 3)  # (1, 768, 8, 14, 14)

        # Trilinear interpolation
        patch_pos_interp = F.interpolate(
            patch_pos,
            size=(T_new, H, W),
            mode='trilinear',
            align_corners=False
        )  # (1, 768, 4, 14, 14)

        # Reshape back to (1, N, C)
        patch_pos_interp = patch_pos_interp.permute(0, 2, 3, 4, 1)  # (1, 4, 14, 14, 768)
        new_pos_embed = patch_pos_interp.reshape(1, T_new * H * W, C)  # (1, 784, 768)

        print(f"[VideoMAE-8Frame]   New pos_embed shape: {new_pos_embed.shape}")

        # Replace positional embeddings
        self.videomae.embeddings.position_embeddings = nn.Parameter(new_pos_embed)
        print(f"[VideoMAE-8Frame]   Positional embeddings interpolated successfully!")

    def _init_weights(self):
        """Initialize the weights of projection layers"""
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.spatial_refine.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VideoMAE with 8 frames

        Args:
            x: Input tensor (B, C, T, H, W) - YOWO format
               B: batch size
               C: channels (3 for RGB)
               T: temporal dimension (8 frames)
               H, W: height and width (224x224)

        Returns:
            Output tensor (B, target_channels, 1, 7, 7) - Matches I3D format
        """
        B, C, T, H, W = x.shape

        # Validate input frames
        if T != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {T}")

        # Reshape for VideoMAE: (B, C, T, H, W) -> (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Forward through VideoMAE encoder
        with torch.amp.autocast('cuda', enabled=False):
            outputs = self.videomae(x, return_dict=True)

        # Extract features
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output  # (B, 768)
        else:
            features = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        # Project to target dimensions
        features = self.projector(features)

        # Reshape to spatial format
        features = features.view(B, self.target_channels,
                                self.target_spatial_size, self.target_spatial_size)

        # Apply spatial refinement with residual
        refined_features = self.spatial_refine(features)
        features = features + refined_features

        # Add temporal dimension
        features = features.unsqueeze(2)  # (B, C, 1, 7, 7)

        return features

    def freeze_encoder(self):
        """Freeze VideoMAE encoder"""
        for param in self.videomae.parameters():
            param.requires_grad = False
        print("[VideoMAE-8Frame] Encoder frozen")

    def unfreeze_encoder(self, last_n_layers: Optional[int] = None):
        """Unfreeze VideoMAE encoder layers"""
        if last_n_layers is None:
            for param in self.videomae.parameters():
                param.requires_grad = True
            print("[VideoMAE-8Frame] All encoder layers unfrozen")
        else:
            encoder_layers = self.videomae.encoder.layer
            num_layers = len(encoder_layers)

            for i, layer in enumerate(encoder_layers):
                if i >= num_layers - last_n_layers:
                    for param in layer.parameters():
                        param.requires_grad = True

            print(f"[VideoMAE-8Frame] Last {last_n_layers} encoder layers unfrozen")

    def get_num_params(self):
        """Get number of parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        videomae_params = sum(p.numel() for p in self.videomae.parameters())
        adapter_params = total - videomae_params

        return {
            'total': total,
            'trainable': trainable,
            'videomae': videomae_params,
            'adapter': adapter_params
        }

    def load_pretrain(self):
        """Compatibility method for YOWOFormer"""
        pass


class VideoMAE8FrameAdvanced(nn.Module):
    """
    Advanced VideoMAE backbone for 8 frames with multiple adapter methods
    Supports: token, cross, hybrid methods
    """

    def __init__(self,
                 model_variant: str = "base",
                 method: str = "cross",
                 pretrained: bool = True,
                 freeze_encoder: bool = True,
                 target_channels: int = 512,
                 target_spatial_size: int = 7,
                 dropout: float = 0.2,
                 num_frames: int = 8):
        super().__init__()

        self.model_map = {
            "base": "MCG-NJU/videomae-base-finetuned-kinetics",
            "base_raw": "MCG-NJU/videomae-base",
            "large": "MCG-NJU/videomae-large-finetuned-kinetics",
        }

        self.model_name = self.model_map.get(model_variant, model_variant)
        self.method = method
        self.target_channels = target_channels
        self.target_spatial_size = target_spatial_size
        self.num_frames = num_frames
        self.pretrained_frames = 16

        self.temporal_tokens_pretrained = self.pretrained_frames // 2
        self.temporal_tokens_target = self.num_frames // 2
        self.spatial_tokens = 14 * 14

        print(f"[VideoMAE-8Frame-Adv] Initializing with {method.upper()} method")
        print(f"[VideoMAE-8Frame-Adv] Model: {self.model_name}")
        print(f"[VideoMAE-8Frame-Adv] Input frames: {num_frames}")
        print(f"[VideoMAE-8Frame-Adv] Target: ({target_channels}, 1, {target_spatial_size}, {target_spatial_size})")

        # Load VideoMAE model
        try:
            self.videomae = VideoMAEModel.from_pretrained(self.model_name)
            self.hidden_size = self.videomae.config.hidden_size
            print(f"[VideoMAE-8Frame-Adv] Model loaded. Hidden size: {self.hidden_size}")
        except Exception as e:
            print(f"[VideoMAE-8Frame-Adv] Error loading model: {e}")
            raise

        # Interpolate positional embeddings
        if num_frames != self.pretrained_frames:
            self._interpolate_pos_embed()

        # Freeze encoder
        if freeze_encoder:
            for param in self.videomae.parameters():
                param.requires_grad = False
            print(f"[VideoMAE-8Frame-Adv] Encoder frozen")

        # Initialize adapter based on method
        # Note: num_tokens for 8 frames = 1 + 4*196 = 785 (vs 1569 for 16 frames)
        num_tokens_8frame = 1 + self.temporal_tokens_target * self.spatial_tokens  # 785

        if method == "token":
            self.adapter = TokenAdapter8Frame(
                hidden_size=self.hidden_size,
                target_channels=target_channels,
                target_spatial_size=target_spatial_size,
                dropout=dropout,
                num_tokens=num_tokens_8frame
            )
        elif method == "cross":
            self.adapter = CrossAttentionAdapter8Frame(
                hidden_size=self.hidden_size,
                target_channels=target_channels,
                target_spatial_size=target_spatial_size,
                dropout=dropout
            )
        elif method == "hybrid":
            self.adapter = HybridAdapter8Frame(
                hidden_size=self.hidden_size,
                target_channels=target_channels,
                target_spatial_size=target_spatial_size,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"[VideoMAE-8Frame-Adv] {method.capitalize()} adapter initialized")

    def _interpolate_pos_embed(self):
        """Interpolate positional embeddings from 16 to 8 frames"""
        print(f"[VideoMAE-8Frame-Adv] Interpolating positional embeddings...")

        pos_embed = self.videomae.embeddings.position_embeddings
        print(f"[VideoMAE-8Frame-Adv]   Original: {pos_embed.shape}")

        # VideoMAE has NO CLS token - all tokens are patch tokens
        patch_pos = pos_embed  # (1, 1568, 768)

        T_orig = self.temporal_tokens_pretrained
        T_new = self.temporal_tokens_target
        H = W = 14
        C = self.hidden_size

        patch_pos = patch_pos.reshape(1, T_orig, H, W, C)
        patch_pos = patch_pos.permute(0, 4, 1, 2, 3)

        patch_pos_interp = F.interpolate(
            patch_pos,
            size=(T_new, H, W),
            mode='trilinear',
            align_corners=False
        )

        patch_pos_interp = patch_pos_interp.permute(0, 2, 3, 4, 1)
        new_pos_embed = patch_pos_interp.reshape(1, T_new * H * W, C)  # (1, 784, 768)

        print(f"[VideoMAE-8Frame-Adv]   New: {new_pos_embed.shape}")

        self.videomae.embeddings.position_embeddings = nn.Parameter(new_pos_embed)
        print(f"[VideoMAE-8Frame-Adv]   Done!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        B, C, T, H, W = x.shape

        if T != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {T}")

        x = x.permute(0, 2, 1, 3, 4)

        with torch.amp.autocast('cuda', enabled=False):
            outputs = self.videomae(x, return_dict=True)

        hidden_states = outputs.last_hidden_state
        spatial_features = self.adapter(hidden_states)

        return spatial_features

    def freeze_encoder(self):
        for param in self.videomae.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self, last_n_layers: Optional[int] = None):
        if last_n_layers is None:
            for param in self.videomae.parameters():
                param.requires_grad = True
        else:
            encoder_layers = self.videomae.encoder.layer
            num_layers = len(encoder_layers)
            for i, layer in enumerate(encoder_layers):
                if i >= num_layers - last_n_layers:
                    for param in layer.parameters():
                        param.requires_grad = True

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        videomae_params = sum(p.numel() for p in self.videomae.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'videomae': videomae_params,
            'adapter': adapter_params,
            'method': self.method
        }

    def load_pretrain(self):
        pass


# Adapter classes for 8-frame version

class TokenAdapter8Frame(nn.Module):
    """Token-based adapter for 8-frame VideoMAE"""

    def __init__(self,
                 hidden_size: int = 768,
                 target_channels: int = 512,
                 target_spatial_size: int = 7,
                 dropout: float = 0.2,
                 num_tokens: int = 785):
        super().__init__()

        self.hidden_size = hidden_size
        self.target_channels = target_channels
        self.target_spatial_size = target_spatial_size
        self.num_tokens = num_tokens
        self.num_spatial_tokens = num_tokens - 1  # 784

        # For 8 frames: 4 temporal * 196 spatial = 784 tokens
        self.patch_size = 14  # spatial: 14x14

        intermediate_dim = hidden_size * 2

        self.token_proj = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, target_channels * 4),
            nn.GELU()
        )

        self.spatial_recon = nn.Sequential(
            nn.Conv2d(target_channels, target_channels, 3, 1, 1, groups=target_channels),
            nn.BatchNorm2d(target_channels),
            nn.GELU(),
            nn.Conv2d(target_channels, target_channels, 1),
            nn.BatchNorm2d(target_channels),
            nn.GELU(),
            nn.Conv2d(target_channels, target_channels, 3, 1, 1),
            nn.BatchNorm2d(target_channels)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((target_spatial_size, target_spatial_size))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]

        # Use spatial tokens (skip CLS token)
        # For 8 frames: 784 spatial tokens
        spatial_tokens = features[:, 1:, :]  # (B, 784, 768)

        # Average pool temporal dimension: (4, 196) -> 196
        # Reshape: (B, 4, 196, 768) -> (B, 196, 768)
        spatial_tokens = spatial_tokens.reshape(B, 4, 196, self.hidden_size)
        spatial_tokens = spatial_tokens.mean(dim=1)  # (B, 196, 768)

        projected = self.token_proj(spatial_tokens)  # (B, 196, C*4)

        # Reshape to spatial: (B, 196, C*4) -> (B, C, 28, 28)
        projected = rearrange(
            projected,
            'b (h w) (c f1 f2) -> b c (h f1) (w f2)',
            h=14, w=14,
            f1=2, f2=2,
            c=self.target_channels
        )

        spatial = self.spatial_recon(projected)
        spatial = self.adaptive_pool(spatial)
        output = spatial.unsqueeze(2)

        return output


class CrossAttentionAdapter8Frame(nn.Module):
    """Cross-attention adapter for 8-frame VideoMAE"""

    def __init__(self,
                 hidden_size: int = 768,
                 target_channels: int = 512,
                 target_spatial_size: int = 7,
                 dropout: float = 0.2,
                 num_heads: int = 8):
        super().__init__()

        self.hidden_size = hidden_size
        self.target_channels = target_channels
        self.target_spatial_size = target_spatial_size
        self.num_heads = num_heads

        # Learnable spatial queries
        self.spatial_queries = nn.Parameter(
            torch.randn(1, target_spatial_size * target_spatial_size, hidden_size)
        )

        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_size, target_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.spatial_refine = nn.Sequential(
            nn.Conv2d(target_channels, target_channels, 3, 1, 1),
            nn.BatchNorm2d(target_channels),
            nn.GELU(),
            nn.Conv2d(target_channels, target_channels, 3, 1, 1),
            nn.BatchNorm2d(target_channels),
            nn.GELU(),
            nn.Conv2d(target_channels, target_channels, 1),
            nn.BatchNorm2d(target_channels)
        )

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        H = W = self.target_spatial_size

        queries = self.spatial_queries.expand(B, -1, -1)

        queries_norm = self.ln1(queries)
        features_norm = self.ln2(features)
        attended, _ = self.cross_attn(queries_norm, features_norm, features_norm)
        queries = queries + attended

        queries_norm = self.ln3(queries)
        refined, _ = self.self_attn(queries_norm, queries_norm, queries_norm)
        queries = queries + refined

        features_proj = self.feature_proj(queries)
        spatial = rearrange(features_proj, 'b (h w) c -> b c h w', h=H, w=W)
        spatial = self.spatial_refine(spatial) + spatial

        output = spatial.unsqueeze(2)
        return output


class HybridAdapter8Frame(nn.Module):
    """Hybrid adapter for 8-frame VideoMAE"""

    def __init__(self,
                 hidden_size: int = 768,
                 target_channels: int = 512,
                 target_spatial_size: int = 7,
                 dropout: float = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.target_channels = target_channels
        self.target_spatial_size = target_spatial_size

        bottleneck_dim = hidden_size // 2
        self.token_bottleneck = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        self.multi_scale_proj = nn.ModuleList([
            nn.Linear(hidden_size, target_channels // 4),
            nn.Linear(hidden_size, target_channels // 2),
            nn.Linear(hidden_size, target_channels // 4),
        ])

        self.conv3d = nn.Sequential(
            nn.Conv3d(target_channels, target_channels,
                     kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(target_channels),
            nn.GELU(),
            nn.Conv3d(target_channels, target_channels,
                     kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(target_channels),
            nn.GELU()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(target_channels, target_channels // 8, 1),
            nn.GELU(),
            nn.Conv2d(target_channels // 8, target_channels, 1),
            nn.Sigmoid()
        )

        self.final_refine = nn.Sequential(
            nn.Conv2d(target_channels, target_channels, 3, 1, 1),
            nn.BatchNorm2d(target_channels),
            nn.GELU(),
            nn.Conv2d(target_channels, target_channels, 1),
            nn.BatchNorm2d(target_channels)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N, D = features.shape
        H = W = self.target_spatial_size

        features = features + self.token_bottleneck(features)

        cls_token = features[:, 0:1, :]
        spatial_tokens = features[:, 1:, :]

        global_feat = cls_token.expand(-1, H*W, -1)

        # For 8 frames: use first 49 tokens or average
        if spatial_tokens.shape[1] >= H*W:
            local_feat = spatial_tokens[:, :H*W, :]
        else:
            # Average pool if not enough tokens
            local_feat = F.adaptive_avg_pool1d(
                spatial_tokens.permute(0, 2, 1), H*W
            ).permute(0, 2, 1)

        combined = 0.7 * local_feat + 0.3 * global_feat

        multi_scale_feats = [proj(combined) for proj in self.multi_scale_proj]
        features_concat = torch.cat(multi_scale_feats, dim=-1)

        spatial = rearrange(features_concat, 'b (h w) c -> b c h w', h=H, w=W)

        spatial_3d = spatial.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        spatial_3d = self.conv3d(spatial_3d)
        spatial = spatial_3d[:, :, 1, :, :]

        attention = self.spatial_attention(spatial)
        spatial = spatial * attention
        spatial = self.final_refine(spatial) + spatial

        output = spatial.unsqueeze(2)
        return output


def build_videomae_8frame(config):
    """
    Build 8-frame VideoMAE backbone from config

    Config example:
    {
        'BACKBONE3D': {
            'TYPE': 'videomae_8frame',
            'VARIANT': 'base',
            'METHOD': 'cross',  # simple, token, cross, hybrid
            'FREEZE': True,
            'TARGET_CHANNELS': 512,
            'TARGET_SPATIAL_SIZE': 7,
            'DROPOUT': 0.2,
            'NUM_FRAMES': 8
        }
    }
    """
    backbone_config = config.get('BACKBONE3D', {})
    method = backbone_config.get('METHOD', 'simple')

    if method == 'simple':
        return VideoMAE8FrameBackbone(
            model_variant=backbone_config.get('VARIANT', 'base'),
            pretrained=True,
            freeze_encoder=backbone_config.get('FREEZE', True),
            target_channels=backbone_config.get('TARGET_CHANNELS', 512),
            target_spatial_size=backbone_config.get('TARGET_SPATIAL_SIZE', 7),
            dropout_rate=backbone_config.get('DROPOUT', 0.2),
            num_frames=backbone_config.get('NUM_FRAMES', 8)
        )
    else:
        return VideoMAE8FrameAdvanced(
            model_variant=backbone_config.get('VARIANT', 'base'),
            method=method,
            pretrained=True,
            freeze_encoder=backbone_config.get('FREEZE', True),
            target_channels=backbone_config.get('TARGET_CHANNELS', 512),
            target_spatial_size=backbone_config.get('TARGET_SPATIAL_SIZE', 7),
            dropout=backbone_config.get('DROPOUT', 0.2),
            num_frames=backbone_config.get('NUM_FRAMES', 8)
        )


# Test function
def test_videomae_8frame():
    """Test the 8-frame VideoMAE backbone"""
    print("="*60)
    print("Testing VideoMAE 8-Frame Backbone")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test simple version
    print("\n[1] Testing Simple 8-Frame Backbone")
    print("-"*40)

    model = VideoMAE8FrameBackbone(
        model_variant="base",
        freeze_encoder=True,
        target_channels=512,
        num_frames=8
    )
    model = model.to(device)
    model.eval()

    # Test input with 8 frames
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 8, 224, 224).to(device)

    print(f"Input shape: {test_input.shape}")
    print(f"Expected output: ({batch_size}, 512, 1, 7, 7)")

    with torch.no_grad():
        output = model(test_input)

    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 512, 1, 7, 7), "Shape mismatch!"
    print("Simple backbone test passed!")

    # Test advanced version with different methods
    methods = ["token", "cross", "hybrid"]
    for method in methods:
        print(f"\n[2] Testing {method.upper()} 8-Frame Backbone")
        print("-"*40)

        model = VideoMAE8FrameAdvanced(
            model_variant="base",
            method=method,
            freeze_encoder=True,
            target_channels=512,
            num_frames=8
        )
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(test_input)

        print(f"Output shape: {output.shape}")
        params = model.get_num_params()
        print(f"Adapter params: {params['adapter']/1e6:.2f}M")
        print(f"{method.upper()} test passed!")

        # Clear memory
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("All 8-Frame VideoMAE tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_videomae_8frame()
