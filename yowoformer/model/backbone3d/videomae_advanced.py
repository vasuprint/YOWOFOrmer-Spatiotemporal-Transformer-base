"""
Advanced VideoMAE Backbone with Multiple Reconstruction Methods
Includes: Token, Cross-attention, and Hybrid methods
For improving from baseline 87.4 mAP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel
import math
from typing import Optional, Tuple, Dict
from einops import rearrange, repeat


class BaseAdapter(nn.Module):
    """Base class for all adapter methods"""

    def __init__(self,
                 hidden_size: int = 768,
                 target_channels: int = 1024,
                 target_spatial_size: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_channels = target_channels
        self.target_spatial_size = target_spatial_size
        self.dropout = dropout

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class TokenAdapter(BaseAdapter):
    """
    Token-based reconstruction method
    Uses spatial tokens to reconstruct spatial features
    Fastest and most memory efficient
    """

    def __init__(self,
                 hidden_size: int = 768,
                 target_channels: int = 1024,
                 target_spatial_size: int = 7,
                 dropout: float = 0.1,
                 num_tokens: int = 197):  # 196 spatial + 1 cls token
        super().__init__(hidden_size, target_channels, target_spatial_size, dropout)

        self.num_tokens = num_tokens
        self.num_spatial_tokens = num_tokens - 1  # Remove CLS token

        # Calculate intermediate dimensions
        self.patch_size = int(math.sqrt(self.num_spatial_tokens))
        intermediate_dim = hidden_size * 2

        # Token to spatial projection
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, target_channels * 4),  # Upsample
            nn.GELU()
        )

        # Spatial reconstruction with depthwise separable convolutions
        self.spatial_recon = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(target_channels, target_channels, 3, 1, 1, groups=target_channels),
            nn.BatchNorm2d(target_channels),
            nn.GELU(),
            # Pointwise conv
            nn.Conv2d(target_channels, target_channels, 1),
            nn.BatchNorm2d(target_channels),
            nn.GELU(),
            # Final adjustment
            nn.Conv2d(target_channels, target_channels, 3, 1, 1),
            nn.BatchNorm2d(target_channels)
        )

        # Adaptive pooling to target size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (target_spatial_size, target_spatial_size)
        )

        self.init_weights()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, D) where N is number of tokens, D is hidden dim
        Returns:
            (B, C, 1, H, W) spatial features
        """
        B = features.shape[0]

        # Use spatial tokens (skip CLS token)
        spatial_tokens = features[:, 1:self.num_spatial_tokens+1, :]  # (B, 196, 768)

        # Project tokens
        projected = self.token_proj(spatial_tokens)  # (B, 196, 1024*4)

        # Reshape to spatial format
        # First reshape to patch grid
        patch_h = patch_w = self.patch_size
        projected = rearrange(
            projected,
            'b (h w) (c f1 f2) -> b c (h f1) (w f2)',
            h=patch_h,
            w=patch_w,
            f1=2,
            f2=2,  # 2x2 upsampling
            c=self.target_channels
        )

        # Apply spatial reconstruction
        spatial = self.spatial_recon(projected)

        # Adaptive pooling to target size
        spatial = self.adaptive_pool(spatial)  # (B, 1024, 7, 7)

        # Add temporal dimension
        output = spatial.unsqueeze(2)  # (B, 1024, 1, 7, 7)

        return output


class SQA(BaseAdapter):
    """
    Cross-attention based reconstruction method
    Uses cross-attention between temporal and learnable spatial queries
    Best quality but more computationally expensive
    """

    def __init__(self,
                 hidden_size: int = 768,
                 target_channels: int = 1024,
                 target_spatial_size: int = 7,
                 dropout: float = 0.1,
                 num_heads: int = 8):
        super().__init__(hidden_size, target_channels, target_spatial_size, dropout)

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Learnable spatial queries
        self.spatial_queries = nn.Parameter(
            torch.randn(1, target_spatial_size * target_spatial_size, hidden_size)
        )

        # Cross-attention layers
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Self-attention for refining spatial features
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_size, target_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Spatial refinement with convolutions
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

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        self.init_weights()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, D) VideoMAE features
        Returns:
            (B, C, 1, H, W) spatial features
        """
        B = features.shape[0]
        H = W = self.target_spatial_size

        # Expand spatial queries for batch
        queries = self.spatial_queries.expand(B, -1, -1)  # (B, 49, 768)

        # Cross-attention: queries attend to VideoMAE features
        queries_norm = self.ln1(queries)
        features_norm = self.ln2(features)
        attended, _ = self.cross_attn(
            queries_norm,
            features_norm,
            features_norm
        )
        queries = queries + attended  # Residual connection

        # Self-attention for spatial coherence
        queries_norm = self.ln3(queries)
        refined, _ = self.self_attn(
            queries_norm,
            queries_norm,
            queries_norm
        )
        queries = queries + refined  # Residual connection

        # Project to target channels
        features_proj = self.feature_proj(queries)  # (B, 49, 1024)

        # Reshape to spatial format
        spatial = rearrange(features_proj, 'b (h w) c -> b c h w', h=H, w=W)

        # Spatial refinement
        spatial = self.spatial_refine(spatial) + spatial  # Residual

        # Add temporal dimension
        output = spatial.unsqueeze(2)  # (B, 1024, 1, 7, 7)

        return output


class HybridAdapter(BaseAdapter):
    """
    Hybrid reconstruction method
    Combines token projection with convolutional refinement
    Balance between speed and quality
    """

    def __init__(self,
                 hidden_size: int = 768,
                 target_channels: int = 1024,
                 target_spatial_size: int = 7,
                 dropout: float = 0.1):
        super().__init__(hidden_size, target_channels, target_spatial_size, dropout)

        # Stage 1: Token projection with bottleneck
        bottleneck_dim = hidden_size // 2
        self.token_bottleneck = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # Stage 2: Multi-scale feature projection
        self.multi_scale_proj = nn.ModuleList([
            nn.Linear(hidden_size, target_channels // 4),  # Small
            nn.Linear(hidden_size, target_channels // 2),  # Medium
            nn.Linear(hidden_size, target_channels // 4),  # Large
        ])

        # Stage 3: 3D convolution for temporal-spatial modeling
        self.conv3d = nn.Sequential(
            nn.Conv3d(target_channels, target_channels,
                     kernel_size=(3, 3, 3),
                     padding=(1, 1, 1)),
            nn.BatchNorm3d(target_channels),
            nn.GELU(),
            nn.Conv3d(target_channels, target_channels,
                     kernel_size=(1, 3, 3),
                     padding=(0, 1, 1)),
            nn.BatchNorm3d(target_channels),
            nn.GELU()
        )

        # Stage 4: Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(target_channels, target_channels // 8, 1),
            nn.GELU(),
            nn.Conv2d(target_channels // 8, target_channels, 1),
            nn.Sigmoid()
        )

        # Stage 5: Final refinement
        self.final_refine = nn.Sequential(
            nn.Conv2d(target_channels, target_channels, 3, 1, 1),
            nn.BatchNorm2d(target_channels),
            nn.GELU(),
            nn.Conv2d(target_channels, target_channels, 1),
            nn.BatchNorm2d(target_channels)
        )

        self.init_weights()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, D) VideoMAE features
        Returns:
            (B, C, 1, H, W) spatial features
        """
        B, N, D = features.shape
        H = W = self.target_spatial_size

        # Stage 1: Bottleneck processing
        features = features + self.token_bottleneck(features)  # (B, N, D)

        # Use both CLS and spatial tokens
        cls_token = features[:, 0:1, :]  # (B, 1, D)
        spatial_tokens = features[:, 1:, :]  # (B, N-1, D)

        # Global feature from CLS token
        global_feat = cls_token.expand(-1, H*W, -1)  # (B, 49, D)

        # Combine global and local features
        combined = 0.7 * spatial_tokens[:, :H*W, :] + 0.3 * global_feat

        # Stage 2: Multi-scale projection
        multi_scale_feats = []
        for proj in self.multi_scale_proj:
            feat = proj(combined)  # (B, 49, C/4 or C/2)
            multi_scale_feats.append(feat)

        # Concatenate multi-scale features
        features_concat = torch.cat(multi_scale_feats, dim=-1)  # (B, 49, C)

        # Reshape to spatial
        spatial = rearrange(features_concat, 'b (h w) c -> b c h w', h=H, w=W)

        # Stage 3: 3D convolution (add temporal dimension first)
        spatial_3d = spatial.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # (B, C, 3, H, W)
        spatial_3d = self.conv3d(spatial_3d)
        spatial = spatial_3d[:, :, 1, :, :]  # Take middle frame (B, C, H, W)

        # Stage 4: Spatial attention
        attention = self.spatial_attention(spatial)
        spatial = spatial * attention

        # Stage 5: Final refinement
        spatial = self.final_refine(spatial) + spatial  # Residual

        # Add temporal dimension
        output = spatial.unsqueeze(2)  # (B, 1024, 1, 7, 7)

        return output


class VideoMAEBackboneAdvanced(nn.Module):
    """
    Advanced VideoMAE backbone with multiple reconstruction methods
    Supports: token, cross, hybrid methods for improved accuracy
    """

    def __init__(self,
                 model_variant: str = "base",
                 method: str = "hybrid",  # token, cross, hybrid
                 pretrained: bool = True,
                 freeze_encoder: bool = True,
                 target_channels: int = 1024,
                 target_spatial_size: int = 7,
                 dropout: float = 0.1):
        super().__init__()

        # Model configuration - All available VideoMAE variants from Hugging Face
        self.model_map = {
            # Small variants (hidden_size=384)
            "small": "MCG-NJU/videomae-small-finetuned-kinetics",
            "small_ssv2": "MCG-NJU/videomae-small-finetuned-ssv2",

            # Base variants (hidden_size=768)
            "base": "MCG-NJU/videomae-base-finetuned-kinetics",
            "base_raw": "MCG-NJU/videomae-base",
            "base_short": "MCG-NJU/videomae-base-short",
            "base_ssv2": "MCG-NJU/videomae-base-finetuned-ssv2",

            # Large variants (hidden_size=1024)
            "large": "MCG-NJU/videomae-large-finetuned-kinetics",
            "large_raw": "MCG-NJU/videomae-large",

            # Huge variants (hidden_size=1280)
            "huge": "MCG-NJU/videomae-huge-finetuned-kinetics",
        }

        self.model_name = self.model_map.get(model_variant, model_variant)
        self.method = method
        self.target_channels = target_channels
        self.target_spatial_size = target_spatial_size

        print(f"[VideoMAE-Advanced] Initializing with {method.upper()} method")
        print(f"[VideoMAE-Advanced] Model: {self.model_name}")
        print(f"[VideoMAE-Advanced] Target: ({target_channels}, 1, {target_spatial_size}, {target_spatial_size})")

        # Load VideoMAE model
        try:
            self.videomae = VideoMAEModel.from_pretrained(self.model_name)
            self.hidden_size = self.videomae.config.hidden_size
            print(f"[VideoMAE-Advanced] Model loaded. Hidden size: {self.hidden_size}")
        except Exception as e:
            print(f"[VideoMAE-Advanced] Error loading model: {e}")
            raise

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.videomae.parameters():
                param.requires_grad = False
            print(f"[VideoMAE-Advanced] Encoder frozen")

        # Initialize adapter based on method
        if method == "token":
            self.adapter = TokenAdapter(
                hidden_size=self.hidden_size,
                target_channels=target_channels,
                target_spatial_size=target_spatial_size,
                dropout=dropout
            )
            print(f"[VideoMAE-Advanced] Token adapter initialized")

        elif method == "cross":
            self.adapter = SQA(
                hidden_size=self.hidden_size,
                target_channels=target_channels,
                target_spatial_size=target_spatial_size,
                dropout=dropout
            )
            print(f"[VideoMAE-Advanced] Cross-attention adapter initialized")

        elif method == "hybrid":
            self.adapter = HybridAdapter(
                hidden_size=self.hidden_size,
                target_channels=target_channels,
                target_spatial_size=target_spatial_size,
                dropout=dropout
            )
            print(f"[VideoMAE-Advanced] Hybrid adapter initialized")

        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) YOWO format input
        Returns:
            (B, C_out, 1, H_out, W_out) spatial features
        """
        B, C, T, H, W = x.shape

        # Convert to VideoMAE format: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Forward through VideoMAE
        with torch.amp.autocast('cuda', enabled=False):
            outputs = self.videomae(x, return_dict=True)

        # Get all hidden states (last layer)
        hidden_states = outputs.last_hidden_state  # (B, N_tokens, D)

        # Apply adapter
        spatial_features = self.adapter(hidden_states)

        return spatial_features

    def freeze_encoder(self):
        """Freeze VideoMAE encoder"""
        for param in self.videomae.parameters():
            param.requires_grad = False
        print(f"[VideoMAE-Advanced] Encoder frozen")

    def unfreeze_encoder(self, last_n_layers: Optional[int] = None):
        """Unfreeze VideoMAE encoder layers"""
        if last_n_layers is None:
            for param in self.videomae.parameters():
                param.requires_grad = True
            print(f"[VideoMAE-Advanced] All encoder layers unfrozen")
        else:
            encoder_layers = self.videomae.encoder.layer
            num_layers = len(encoder_layers)

            for i, layer in enumerate(encoder_layers):
                if i >= num_layers - last_n_layers:
                    for param in layer.parameters():
                        param.requires_grad = True

            print(f"[VideoMAE-Advanced] Last {last_n_layers} encoder layers unfrozen")

    def get_num_params(self):
        """Get parameter count"""
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
        """Compatibility method for YOWOFormer"""
        pass  # VideoMAE weights loaded in __init__


def build_videomae_advanced(config):
    """
    Build advanced VideoMAE backbone from config

    Config example:
    {
        'BACKBONE3D': {
            'TYPE': 'videomae',
            'VARIANT': 'base',
            'METHOD': 'hybrid',  # token, cross, hybrid
            'FREEZE': True,
            'TARGET_CHANNELS': 1024,
            'TARGET_SPATIAL_SIZE': 7,
            'DROPOUT': 0.1
        }
    }
    """
    backbone_config = config.get('BACKBONE3D', {})

    return VideoMAEBackboneAdvanced(
        model_variant=backbone_config.get('VARIANT', 'base'),
        method=backbone_config.get('METHOD', 'hybrid'),
        pretrained=True,
        freeze_encoder=backbone_config.get('FREEZE', True),
        target_channels=backbone_config.get('TARGET_CHANNELS', 1024),
        target_spatial_size=backbone_config.get('TARGET_SPATIAL_SIZE', 7),
        dropout=backbone_config.get('DROPOUT', 0.1)
    )


# Test function
if __name__ == "__main__":
    print("="*60)
    print("Testing Advanced VideoMAE Backbone")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test all three methods
    methods = ["token", "cross", "hybrid"]

    for method in methods:
        print(f"\n🔍 Testing {method.upper()} method")
        print("-"*40)

        # Create model
        model = VideoMAEBackboneAdvanced(
            model_variant="base",
            method=method,
            freeze_encoder=True
        )
        model = model.to(device)
        model.eval()

        # Test input
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 16, 224, 224).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(test_input)

        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")

        # Check parameters
        params = model.get_num_params()
        print(f"  Adapter params: {params['adapter']/1e6:.2f}M")
        print(f"  Trainable params: {params['trainable']/1e6:.2f}M")

        # Memory usage
        if device.type == 'cuda':
            memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
            print(f"  GPU Memory: {memory_mb:.1f} MB")
            torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("✅ All methods tested successfully!")
    print("="*60)