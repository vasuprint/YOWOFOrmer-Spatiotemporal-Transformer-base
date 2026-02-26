"""
YOLOv11 Backbone Implementation for YOWOFormer
ใช้ Ultralytics YOLOv11 เป็น 2D Backbone สำหรับ spatiotemporal action detection
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path


class YOLOv11Backbone(nn.Module):
    """
    YOLOv11 Backbone with multi-scale feature extraction
    Compatible with YOWOFormer architecture
    """

    def __init__(self,
                 version='n',  # n, s, m, l, x
                 pretrained=True,
                 pretrained_path=None,  # Custom pretrained weights path
                 freeze_backbone=False):
        """
        Initialize YOLOv11 backbone

        Args:
            version: Model version ('n', 's', 'm', 'l', 'x')
            pretrained: Use pretrained weights
            pretrained_path: Path to custom pretrained weights (optional)
            freeze_backbone: Freeze backbone parameters
        """
        super(YOLOv11Backbone, self).__init__()

        # Model configuration
        self.version = version

        # Determine model path
        if pretrained_path and Path(pretrained_path).exists():
            # Use custom pretrained weights
            model_path = pretrained_path
            print(f"Loading YOLOv11{version} from custom path: {model_path}")
        elif pretrained:
            # Use default pretrained weights
            model_path = f"yolo11{version}.pt"
            print(f"Loading pretrained YOLOv11{version} from {model_path}")
        else:
            # Load model architecture only
            model_path = f"yolo11{version}.yaml"
            print(f"Creating YOLOv11{version} from scratch (no pretrained weights)")

        # Load YOLOv11 model
        yolo_model = YOLO(model_path)
        model = yolo_model.model.model

        # Extract backbone and neck layers (up to layer 16 for YOWOFormer)
        self.backbone = nn.ModuleList(list(model.children())[:17])

        # Layers to save for concatenation
        self.intermediate_layers = [4, 6, 10]  # เก็บ intermediate outputs
        self.feature_layers = [10, 13, 16]  # เลเยอร์ที่ต้องการ Feature Maps

        # Identify concat layers (they have 'd' attribute)
        self.concat_layers = []
        for idx, layer in enumerate(self.backbone):
            if hasattr(layer, 'd'):  # This is a Concat layer
                self.concat_layers.append(idx)

        # Set output channels based on version
        self._set_output_channels()

        # Freeze if requested
        if freeze_backbone:
            self.freeze()

    def _set_output_channels(self):
        """Set output channel dimensions based on model version"""
        channel_config = {
            'n': [64, 128, 256],    # nano
            's': [128, 256, 512],   # small
            'm': [192, 384, 768],   # medium
            'l': [256, 512, 1024],  # large
            'x': [320, 640, 1280]   # extra large
        }
        self.out_channels = channel_config.get(self.version, [64, 128, 256])

    def forward(self, x):
        """
        Forward pass through YOLOv11 backbone

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            List of feature maps from layers [10, 13, 16]
        """
        intermediates = {}
        features = {}

        for i, layer in enumerate(self.backbone):
            # Handle Concat layers
            if i == 12:  # Concat layer 12
                # Layer 12 = concat(11, 6)
                if 6 in intermediates:
                    x = layer([x, intermediates[6]])
                else:
                    x = layer([x])
            elif i == 15:  # Concat layer 15
                # Layer 15 = concat(14, 4)
                if 4 in intermediates:
                    x = layer([x, intermediates[4]])
                else:
                    x = layer([x])
            elif i in self.concat_layers:
                # Other concat layers - just pass through
                x = layer([x])
            else:
                # Regular layers
                x = layer(x)

            # Save intermediate outputs
            if i in self.intermediate_layers:
                intermediates[i] = x

            # Save feature outputs
            if i in self.feature_layers:
                features[i] = x

        # Return features in order [10, 13, 16]
        return [features[i] for i in sorted(self.feature_layers)]

    def freeze(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"YOLOv11{self.version} backbone frozen")

    def unfreeze(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"YOLOv11{self.version} backbone unfrozen")

    def load_pretrain(self):
        """Compatibility method for YOWOFormer - weights already loaded in __init__"""
        pass


# Alias for compatibility
YOLOv11FeatureExtractor = YOLOv11Backbone


# Helper function for YOWOFormer compatibility
def build_yolov11_backbone(config):
    """
    Build YOLOv11 backbone from config

    Args:
        config: Configuration dictionary with options:
            - backbone2D: Name like 'yolov11_n'
            - pretrain_2d: Boolean or path to weights
            - pretrained_path_2d: Path to custom pretrained weights
            - freeze_backbone2d: Freeze backbone

    Returns:
        YOLOv11Backbone instance
    """
    # Parse configuration
    backbone_name = config.get('backbone2D', 'yolov11_n')

    # Extract version (n, s, m, l, x)
    if '_' in backbone_name:
        version = backbone_name.split('_')[-1]
    else:
        version = 'n'  # default to nano

    # Handle pretrained weights
    pretrain_2d = config.get('pretrain_2d', True)
    pretrained_path = None

    # Check for custom pretrained path
    if isinstance(pretrain_2d, str):
        # pretrain_2d is a path
        pretrained_path = pretrain_2d
        pretrained = True
    else:
        # Check for separate pretrained_path_2d field
        pretrained_path = config.get('pretrained_path_2d', None)
        pretrained = pretrain_2d

    # Also check for pretrain_path_2d (alternative naming)
    if not pretrained_path:
        pretrained_path = config.get('pretrain_path_2d', None)

    freeze = config.get('freeze_backbone2d', False)

    # Also check alternative naming
    if not freeze:
        freeze = config.get('freeze_bb2D', False)

    # Create model
    model = YOLOv11Backbone(
        version=version,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        freeze_backbone=freeze
    )

    print(f"✅ YOLOv11{version} backbone created for YOWOFormer")
    print(f"   Output channels: {model.out_channels}")
    print(f"   Feature layers: {model.feature_layers}")
    if pretrained_path:
        print(f"   Pretrained weights: {pretrained_path}")
    else:
        print(f"   Pretrained: {pretrained}")
    print(f"   Frozen: {freeze}")

    return model


# Alternative name for compatibility
build_yolov11 = build_yolov11_backbone


# Test function
if __name__ == "__main__":
    print("="*60)
    print("Testing YOLOv11 Backbone for YOWOFormer")
    print("="*60)

    # Test different versions
    for version in ['n', 's']:
        print(f"\n🔍 Testing YOLOv11{version}")
        print("-"*40)

        # Create model
        model = YOLOv11Backbone(version=version, pretrained=False)

        # Test with YOWOFormer default input size
        batch_size = 2
        img_size = 224  # YOWOFormer default
        x = torch.randn(batch_size, 3, img_size, img_size)

        # Forward pass
        features = model(x)

        # Print results
        print(f"Input shape: {x.shape}")
        for i, feat in enumerate(features):
            scale = 2**(i+3)  # 8, 16, 32
            print(f"P{i+3} (stride {scale}): {feat.shape}")

        # Count parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total:,} total, {trainable:,} trainable")

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)