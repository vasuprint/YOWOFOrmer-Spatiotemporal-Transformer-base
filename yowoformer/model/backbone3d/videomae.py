"""
VideoMAE Backbone Adapter for YOWOFormer
Phase 1: Simple Projection Method for UCF101-24
Compatible with RTX 3090 (24GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel, AutoImageProcessor
import numpy as np
from typing import Optional, Tuple, Dict


class VideoMAEBackbone(nn.Module):
    """
    VideoMAE backbone wrapper for YOWO
    Phase 1: Using simple projection for quick testing
    """

    def __init__(self,
                 model_variant: str = "base",
                 pretrained: bool = True,
                 freeze_encoder: bool = True,  # Save memory for RTX 3090
                 target_channels: int = 1024,  # Match I3D output
                 target_spatial_size: int = 7,  # Match I3D spatial
                 dropout_rate: float = 0.2):

        super().__init__()

        # Model selection - All available VideoMAE variants from Hugging Face
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
        self.target_channels = target_channels
        self.target_spatial_size = target_spatial_size

        print(f"[VideoMAE] Initializing VideoMAE backbone")
        print(f"[VideoMAE] Model: {self.model_name}")
        print(f"[VideoMAE] Method: Simple Projection (Phase 1)")
        print(f"[VideoMAE] Target output: ({target_channels}, 1, {target_spatial_size}, {target_spatial_size})")

        # Load VideoMAE model
        try:
            self.videomae = VideoMAEModel.from_pretrained(self.model_name)
            self.hidden_size = self.videomae.config.hidden_size  # 768 for base
            print(f"[VideoMAE] Model loaded successfully. Hidden size: {self.hidden_size}")
        except Exception as e:
            print(f"[VideoMAE] Error loading model: {e}")
            raise

        # Freeze encoder to save memory
        if freeze_encoder:
            for param in self.videomae.parameters():
                param.requires_grad = False
            print(f"[VideoMAE] Encoder frozen (saving ~350MB gradient memory)")

        # Simple Projection Method (Phase 1)
        # This is the simplest approach to test if VideoMAE works
        self.projector = nn.Sequential(
            # Project from VideoMAE hidden size to larger space
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            # Project to target spatial dimensions
            nn.Linear(self.hidden_size * 2, target_channels * target_spatial_size * target_spatial_size),
            nn.LayerNorm(target_channels * target_spatial_size * target_spatial_size)
        )

        # Spatial refinement with convolutions
        # This helps create spatial coherence in the projected features
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

        # Initialize weights
        self._init_weights()

        print(f"[VideoMAE] Adapter layers initialized")

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
        Forward pass through VideoMAE with simple projection

        Args:
            x: Input tensor (B, C, T, H, W) - YOWO format
               B: batch size
               C: channels (3 for RGB)
               T: temporal dimension (16 frames)
               H, W: height and width (224x224)

        Returns:
            Output tensor (B, 1024, 1, 7, 7) - Matches I3D format
        """
        B, C, T, H, W = x.shape

        # Reshape for VideoMAE: (B, C, T, H, W) -> (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Forward through VideoMAE encoder
        with torch.amp.autocast('cuda', enabled=False):  # Disable for stability
            outputs = self.videomae(x, return_dict=True)

        # Extract features
        # VideoMAE returns pooled features or we use CLS token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output  # (B, 768)
        else:
            # Use CLS token (first token)
            features = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        # Simple projection to target dimensions
        features = self.projector(features)  # (B, 1024 * 7 * 7)

        # Reshape to spatial format
        features = features.view(B, self.target_channels,
                                self.target_spatial_size, self.target_spatial_size)
        # features shape: (B, 1024, 7, 7)

        # Apply spatial refinement with residual connection
        refined_features = self.spatial_refine(features)
        features = features + refined_features  # Residual connection

        # Add temporal dimension (D=1) to match I3D output format
        # I3D outputs: (B, C, D, H, W) where D=1
        features = features.unsqueeze(2)  # (B, 1024, 1, 7, 7)

        return features

    def freeze_encoder(self):
        """Freeze VideoMAE encoder"""
        for param in self.videomae.parameters():
            param.requires_grad = False
        print("[VideoMAE] Encoder frozen")

    def unfreeze_encoder(self, last_n_layers: Optional[int] = None):
        """
        Unfreeze VideoMAE encoder
        Args:
            last_n_layers: If specified, only unfreeze last n layers
        """
        if last_n_layers is None:
            # Unfreeze all
            for param in self.videomae.parameters():
                param.requires_grad = True
            print("[VideoMAE] All encoder layers unfrozen")
        else:
            # Unfreeze only last n layers
            encoder_layers = self.videomae.encoder.layer
            num_layers = len(encoder_layers)

            for i, layer in enumerate(encoder_layers):
                if i >= num_layers - last_n_layers:
                    for param in layer.parameters():
                        param.requires_grad = True

            print(f"[VideoMAE] Last {last_n_layers} encoder layers unfrozen")

    def get_num_params(self):
        """Get number of parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Breakdown
        videomae_params = sum(p.numel() for p in self.videomae.parameters())
        adapter_params = total - videomae_params

        return {
            'total': total,
            'trainable': trainable,
            'videomae': videomae_params,
            'adapter': adapter_params
        }

    def load_pretrain(self):
        """
        Compatibility method for YOWOFormer
        VideoMAE weights are already loaded in __init__ from Hugging Face
        This method exists only for compatibility with YOWOFormer's initialization flow
        """
        # VideoMAE automatically loads pretrained weights from Hugging Face
        # when using VideoMAEModel.from_pretrained() in __init__
        # So we don't need to do anything here
        pass


def build_videomae_backbone(config):
    """
    Build VideoMAE backbone from config

    Config example:
    {
        'BACKBONE3D': {
            'TYPE': 'videomae',
            'VARIANT': 'base',  # base or large
            'FREEZE': True,     # Freeze encoder initially
            'TARGET_CHANNELS': 1024,
            'TARGET_SPATIAL_SIZE': 7,
            'DROPOUT': 0.2
        }
    }
    """
    backbone_config = config.get('BACKBONE3D', {})

    return VideoMAEBackbone(
        model_variant=backbone_config.get('VARIANT', 'base'),
        pretrained=True,
        freeze_encoder=backbone_config.get('FREEZE', True),
        target_channels=backbone_config.get('TARGET_CHANNELS', 1024),
        target_spatial_size=backbone_config.get('TARGET_SPATIAL_SIZE', 7),
        dropout_rate=backbone_config.get('DROPOUT', 0.2)
    )


# Test function
def test_videomae_backbone():
    """Test the VideoMAE backbone independently"""
    print("="*60)
    print("Testing VideoMAE Backbone for YOWO (Simple Projection)")
    print("="*60)

    # Create model
    model = VideoMAEBackbone(
        model_variant="base",
        freeze_encoder=True
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Test input (YOWO format)
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 16, 224, 224).to(device)

    print(f"\nTest Configuration:")
    print(f"  Device: {device}")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Expected output: ({batch_size}, 1024, 1, 7, 7)")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(test_input)

    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Check shape
    expected_shape = (batch_size, 1024, 1, 7, 7)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print(f"✓ Output shape matches I3D format!")

    # Count parameters
    params = model.get_num_params()
    print(f"\nModel Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  VideoMAE encoder: {params['videomae']:,}")
    print(f"  Adapter layers: {params['adapter']:,}")

    # Memory estimation
    param_memory_mb = params['total'] * 4 / (1024**2)  # fp32
    param_memory_fp16_mb = params['total'] * 2 / (1024**2)  # fp16

    print(f"\nMemory Estimation:")
    print(f"  Model weights (fp32): {param_memory_mb:.1f} MB")
    print(f"  Model weights (fp16): {param_memory_fp16_mb:.1f} MB")

    # Estimate for batch size 8
    batch8_memory_gb = (param_memory_mb + 8 * 16 * 3 * 224 * 224 * 4 / (1024**2)) / 1024
    print(f"  Estimated for batch_size=8: ~{batch8_memory_gb:.1f} GB")

    print("\n" + "="*60)
    print("✓ VideoMAE backbone ready for YOWO integration!")
    print("="*60)


if __name__ == "__main__":
    test_videomae_backbone()