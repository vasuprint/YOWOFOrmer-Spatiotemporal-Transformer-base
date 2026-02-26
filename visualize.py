"""
YOWOFormer Visualization Tool
=============================
All-in-one visualization: GradCAM, Attention Map, BBox Detection
Supports AVA Dataset frames and video files (.mp4/.avi)
CLI + Interactive mode

Usage:
    # Interactive mode
    python visualize.py

    # CLI mode (AVA)
    python visualize.py --checkpoint weights/best.pth \
        --input-type ava --video-id 9F2voT6QWvQ --timestamp 940 \
        --mode all --save output/ --show

    # CLI mode (Video file)
    python visualize.py --checkpoint weights/best.pth \
        --input-type video --video path/to/video.mp4 --frame 100 \
        --mode gradcam attention --save output/
"""

import sys
import os
import re
import argparse
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ============================================================
# Constants
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AVA_FRAMES_DEFAULT = "data/AVA_Dataset/frames"
DEFAULT_CHECKPOINT = "weight_Table/T3/ava_fair_yolov11_crossattn_[compleate]/best.pth"

# AVA v2.1 action labels — index = AVA_ID - 1 (80 slots, some unused)
AVA_LABELS = [
    "bend/bow",          # ID 1  → idx 0
    "",                  # ID 2  → idx 1  (unused)
    "crouch/kneel",      # ID 3  → idx 2
    "dance",             # ID 4  → idx 3
    "fall down",         # ID 5  → idx 4
    "get up",            # ID 6  → idx 5
    "jump/leap",         # ID 7  → idx 6
    "lie/sleep",         # ID 8  → idx 7
    "martial art",       # ID 9  → idx 8
    "run/jog",           # ID 10 → idx 9
    "sit",               # ID 11 → idx 10
    "stand",             # ID 12 → idx 11
    "swim",              # ID 13 → idx 12
    "walk",              # ID 14 → idx 13
    "answer phone",      # ID 15 → idx 14
    "",                  # ID 16 → idx 15 (unused)
    "carry/hold",        # ID 17 → idx 16
    "",                  # ID 18 → idx 17 (unused)
    "",                  # ID 19 → idx 18 (unused)
    "climb",             # ID 20 → idx 19
    "",                  # ID 21 → idx 20 (unused)
    "close",             # ID 22 → idx 21
    "",                  # ID 23 → idx 22 (unused)
    "cut",               # ID 24 → idx 23
    "",                  # ID 25 → idx 24 (unused)
    "dress/put on",      # ID 26 → idx 25
    "drink",             # ID 27 → idx 26
    "drive",             # ID 28 → idx 27
    "eat",               # ID 29 → idx 28
    "enter",             # ID 30 → idx 29
    "",                  # ID 31 → idx 30 (unused)
    "",                  # ID 32 → idx 31 (unused)
    "",                  # ID 33 → idx 32 (unused)
    "hit (object)",      # ID 34 → idx 33
    "",                  # ID 35 → idx 34 (unused)
    "lift/pick up",      # ID 36 → idx 35
    "listen",            # ID 37 → idx 36
    "open",              # ID 38 → idx 37
    "",                  # ID 39 → idx 38 (unused)
    "",                  # ID 40 → idx 39 (unused)
    "play instrument",   # ID 41 → idx 40
    "",                  # ID 42 → idx 41 (unused)
    "point to",          # ID 43 → idx 42
    "",                  # ID 44 → idx 43 (unused)
    "pull",              # ID 45 → idx 44
    "push (object)",     # ID 46 → idx 45
    "put down",          # ID 47 → idx 46
    "read",              # ID 48 → idx 47
    "ride",              # ID 49 → idx 48
    "",                  # ID 50 → idx 49 (unused)
    "sail boat",         # ID 51 → idx 50
    "shoot",             # ID 52 → idx 51
    "",                  # ID 53 → idx 52 (unused)
    "smoke",             # ID 54 → idx 53
    "",                  # ID 55 → idx 54 (unused)
    "take a photo",      # ID 56 → idx 55
    "text/cellphone",    # ID 57 → idx 56
    "throw",             # ID 58 → idx 57
    "touch",             # ID 59 → idx 58
    "turn",              # ID 60 → idx 59
    "watch (TV)",        # ID 61 → idx 60
    "work on computer",  # ID 62 → idx 61
    "write",             # ID 63 → idx 62
    "fight/hit (person)",# ID 64 → idx 63
    "give/serve",        # ID 65 → idx 64
    "grab (person)",     # ID 66 → idx 65
    "hand clap",         # ID 67 → idx 66
    "hand shake",        # ID 68 → idx 67
    "hand wave",         # ID 69 → idx 68
    "hug",               # ID 70 → idx 69
    "",                  # ID 71 → idx 70 (unused)
    "kiss",              # ID 72 → idx 71
    "lift (person)",     # ID 73 → idx 72
    "listen to (person)",# ID 74 → idx 73
    "",                  # ID 75 → idx 74 (unused)
    "push (person)",     # ID 76 → idx 75
    "sing",              # ID 77 → idx 76
    "take from (person)",# ID 78 → idx 77
    "talk to",           # ID 79 → idx 78
    "watch (person)",    # ID 80 → idx 79
]

# UCF101-24 action labels (24 classes)
UCF_LABELS = [
    "Basketball", "BasketballDunk", "Biking", "CliffDiving", "CricketBowling",
    "Diving", "Fencing", "FloorGymnastics", "GolfSwing", "HorseRiding",
    "IceDancing", "LongJump", "PoleVault", "RopeClimbing", "SalsaSpin",
    "SkateBoarding", "Skiing", "Skijet", "SoccerJuggling", "Surfing",
    "TennisSwing", "TrampolineJumping", "VolleyballSpiking", "WalkingWithDog",
]

# Box colors for visualization
BOX_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
    "#00FFFF", "#FF8000", "#8000FF", "#00FF80", "#FF0080",
]


# ============================================================
# Input Loaders
# ============================================================
def load_clip_ava(video_id, timestamp, frames_root, clip_length=16, img_size=224):
    """Load a clip from AVA dataset frames."""
    key_frame_idx = (timestamp - 900) * 30 + 1

    frames = []
    for i in reversed(range(clip_length)):
        fidx = max(1, key_frame_idx - i)
        fpath = Path(frames_root) / video_id / f"{video_id}_{fidx:06d}.jpg"
        if not fpath.exists():
            fpath = Path(frames_root) / video_id / f"{video_id}_{key_frame_idx:06d}.jpg"
        if not fpath.exists():
            raise FileNotFoundError(f"Frame not found: {fpath}")
        img = Image.open(fpath).convert("RGB").resize((img_size, img_size))
        frames.append(img)

    key_frame = frames[-1].copy()
    clip = _frames_to_tensor(frames, img_size)
    return clip, key_frame


def load_clip_video(video_path, frame_idx, clip_length=16, img_size=224):
    """Load a clip from a video file (.mp4/.avi)."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for video loading. Install: pip install opencv-python")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total_frames:
        raise ValueError(f"Frame {frame_idx} exceeds total frames ({total_frames})")

    start_idx = max(0, frame_idx - clip_length + 1)
    frames = []

    for i in range(start_idx, frame_idx + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            if frames:
                frame = frames[-1]
            else:
                raise ValueError(f"Cannot read frame {i}")
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).resize((img_size, img_size))
        frames.append(frame)

    cap.release()

    # Pad if fewer than clip_length frames
    while len(frames) < clip_length:
        frames.insert(0, frames[0])

    key_frame = frames[-1].copy()
    clip = _frames_to_tensor(frames, img_size)
    return clip, key_frame


def _frames_to_tensor(frames, img_size):
    """Convert list of PIL images to normalized tensor [1, 3, T, H, W]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensors = []
    for f in frames:
        if isinstance(f, np.ndarray):
            t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
        else:
            t = torch.from_numpy(np.array(f)).permute(2, 0, 1).float() / 255.0
        t = (t - mean) / std
        tensors.append(t)
    clip = torch.stack(tensors, dim=1)  # [3, T, H, W]
    return clip.unsqueeze(0)  # [1, 3, T, H, W]


# ============================================================
# Model Loading
# ============================================================
def load_model(checkpoint_path, device=DEVICE):
    """Load YOWOFormer model from checkpoint. Auto-detect config."""
    from yowoformer.model.yowoformer import build_yowoformer

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    # Auto-detect videomae method from config
    method = cfg.get('videomae_method', None)
    if method is None:
        bb3d_cfg = cfg.get('BACKBONE3D', {})
        method = bb3d_cfg.get('METHOD', 'simple')

    # Map method names
    method_map = {"simple": "simple", "token": "token", "cross": "cross", "hybrid": "hybrid"}
    if method in method_map:
        method = method_map[method]

    clip_length = cfg.get('clip_length', 16)

    bb2d = cfg.get('backbone2D', 'yolov11_n')

    config = {
        'dataset': cfg.get('dataset', 'ava'),
        'data_root': cfg.get('data_root', 'data/AVA_Dataset'),
        'num_classes': cfg.get('num_classes', 80),
        'backbone2D': bb2d,
        'backbone3D': 'videomae',
        'videomae_variant': cfg.get('videomae_variant', 'base'),
        'videomae_method': method,
        'fusion_module': cfg.get('fusion_module', 'Simple'),
        'mode': cfg.get('mode', 'decoupled'),
        'interchannels': cfg.get('interchannels', [256, 256, 256]),
        'img_size': cfg.get('img_size', IMG_SIZE),
        'clip_length': clip_length,
        'sampling_rate': cfg.get('sampling_rate', 1),
        'pretrain_2d': False,
        'pretrain_3d': False,
        'freeze_bb2D': False,
        'freeze_bb3D': False,
        'pretrain_path': None,
        'active_checker': False,
        'BACKBONE3D': cfg.get('BACKBONE3D', {
            'TYPE': 'videomae',
            'VARIANT': cfg.get('videomae_variant', 'base'),
            'METHOD': method,
            'FREEZE': False,
            'TARGET_CHANNELS': 1024,
            'TARGET_SPATIAL_SIZE': 7,
            'DROPOUT': 0.2,
        }),
        'LOSS': cfg.get('LOSS', {
            'TAL': {
                'top_k': 10, 'alpha': 1.0, 'beta': 6.0, 'radius': 2.5,
                'iou_type': 'ciou', 'iou_weight': 2.0,
                'scale_cls_loss': 0.5, 'scale_box_loss': 7.5, 'scale_dfl_loss': 1.5,
                'soft_label': True
            }
        }),
    }

    # YOLOv8 needs BACKBONE2D config
    if bb2d == 'yolov8' and 'BACKBONE2D' in cfg:
        config['BACKBONE2D'] = cfg['BACKBONE2D']
        # Fix pretrain path to avoid FileNotFoundError
        if 'YOLOv8' in config['BACKBONE2D']:
            config['BACKBONE2D']['YOLOv8']['PRETRAIN'] = {
                k: None for k in config['BACKBONE2D']['YOLOv8'].get('PRETRAIN', {})
            }

    model = build_yowoformer(config)

    # Load weights
    state_dict = ckpt['model_state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [Model] Missing keys: {len(missing)}")
    if unexpected:
        print(f"  [Model] Unexpected keys: {len(unexpected)}")

    model.eval()
    model.to(device)

    info = {
        'method': method,
        'fusion': cfg.get('fusion_module', 'Simple'),
        'dataset': cfg.get('dataset', 'ava'),
        'num_classes': cfg.get('num_classes', 80),
        'clip_length': clip_length,
        'backbone2D': cfg.get('backbone2D', 'yolov11_n'),
        'videomae_variant': cfg.get('videomae_variant', 'base'),
    }

    return model, info


# ============================================================
# GradCAM
# ============================================================
class GradCAM:
    """Grad-CAM with flexible target layer. Uses absolute value for robust maps."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, input, output):
        self.activations = output.detach() if isinstance(output, torch.Tensor) else output[0].detach()

    def _save_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, clip, img_size=IMG_SIZE):
        self.model.zero_grad()
        clip = clip.requires_grad_(True)

        # Use training mode for raw logits (eval sigmoid causes vanishing gradients)
        was_training = self.model.training
        self.model.train()
        output = self.model(clip)
        if not was_training:
            self.model.eval()

        # Parse training output: list of [B, C, H, W] per scale
        if isinstance(output, (list, tuple)):
            preds = []
            for o in output:
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    preds.append(o.flatten(2).permute(0, 2, 1))
                elif isinstance(o, torch.Tensor) and o.dim() == 3:
                    preds.append(o)
            if preds:
                output = torch.cat(preds, dim=1)

        # Extract class logits and backprop from top-k
        if output.dim() == 3:
            num_cls = self.model.detection_head.nc
            cls_logits = output[0, :, -num_cls:]
        elif output.dim() == 2:
            cls_logits = output[:, -80:]
        else:
            cls_logits = output.flatten()

        topk_vals = cls_logits.flatten().topk(k=min(200, cls_logits.numel())).values
        score = topk_vals.sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            print("  WARNING: No activations/gradients captured!")
            return np.zeros((img_size, img_size))

        act = self.activations
        grad = self.gradients
        if act.dim() == 5:
            act = act.squeeze(2)
        if grad.dim() == 5:
            grad = grad.squeeze(2)

        if act.dim() == 4:  # [B, C, H, W]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
        elif act.dim() == 3:  # [B, N, D] token-based
            weights = grad.mean(dim=1, keepdim=True)
            cam = (weights * act).sum(dim=2)
            n = cam.shape[1]
            h = w = int(n ** 0.5)
            if h * w != n:
                h, w = 7, 7
                cam = cam[:, :h * w]
            cam = cam.view(1, 1, h, w)
        else:
            return np.zeros((img_size, img_size))

        cam = cam.abs()
        cam = F.interpolate(cam, size=(img_size, img_size), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


class MultiScaleGradCAM:
    """Grad-CAM hooking at ALL fusion cls scales, merged into one heatmap."""

    def __init__(self, model, layers):
        self.model = model
        self.scale_acts = {}
        self.scale_grads = {}
        self.handles = []
        for i, layer in enumerate(layers):
            self._register(i, layer)

    def _register(self, idx, layer):
        def fwd(m, inp, out, i=idx):
            self.scale_acts[i] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
        def bwd(m, gi, go, i=idx):
            self.scale_grads[i] = go[0].detach()
        self.handles.append(layer.register_forward_hook(fwd))
        self.handles.append(layer.register_full_backward_hook(bwd))

    def generate(self, clip, img_size=IMG_SIZE):
        self.model.zero_grad()
        clip = clip.requires_grad_(True)

        was_training = self.model.training
        self.model.train()
        output = self.model(clip)
        if not was_training:
            self.model.eval()

        if isinstance(output, (list, tuple)):
            preds = []
            for o in output:
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    preds.append(o.flatten(2).permute(0, 2, 1))
                elif isinstance(o, torch.Tensor) and o.dim() == 3:
                    preds.append(o)
            if preds:
                output = torch.cat(preds, dim=1)

        if output.dim() == 3:
            num_cls = self.model.detection_head.nc
            cls_logits = output[0, :, -num_cls:]
        else:
            cls_logits = output.flatten()

        topk_vals = cls_logits.flatten().topk(k=min(200, cls_logits.numel())).values
        score = topk_vals.sum()
        score.backward(retain_graph=False)

        cams = []
        for i in sorted(self.scale_acts.keys()):
            act = self.scale_acts[i]
            grad = self.scale_grads.get(i)
            if grad is None:
                continue
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = cam.abs()
            cam = F.interpolate(cam, size=(img_size, img_size), mode='bilinear', align_corners=False)
            cams.append(cam)

        if not cams:
            return np.zeros((img_size, img_size))

        merged = torch.stack(cams).max(dim=0).values
        merged = merged.squeeze().cpu().numpy()
        if merged.max() > 0:
            merged = (merged - merged.min()) / (merged.max() - merged.min())
        return merged

    def remove(self):
        for h in self.handles:
            h.remove()


def get_gradcam_target_layer(model, method, target='adapter'):
    """Get target layer for GradCAM based on adapter type and target location."""
    if target == 'adapter':
        net3d = model.net3D
        if method == 'simple':
            return net3d.spatial_refine
        elif method == 'token':
            return net3d.adapter.spatial_recon
        elif method == 'cross':
            return net3d.adapter.spatial_refine
        elif method == 'hybrid':
            return net3d.adapter.final_refine
        else:
            raise ValueError(f"Unknown method: {method}")
    elif target == 'fusion':
        # Return list of fusion cls layers for MultiScaleGradCAM
        if hasattr(model.fusion, 'cls'):
            return list(model.fusion.cls)
        elif hasattr(model.fusion, 'cls_blocks'):
            return list(model.fusion.cls_blocks)
        else:
            # Fallback: try to find output convs
            return [model.fusion]
    elif target == 'detection':
        return list(model.detection_head.cls)
    else:
        raise ValueError(f"Unknown target: {target}")


def run_gradcam(model, clip, method, target='adapter', img_size=IMG_SIZE):
    """Run GradCAM and return heatmap numpy array [H, W] in [0, 1]."""
    layer_or_layers = get_gradcam_target_layer(model, method, target)

    if isinstance(layer_or_layers, list):
        gc = MultiScaleGradCAM(model, layer_or_layers)
    else:
        gc = GradCAM(model, layer_or_layers)

    cam = gc.generate(clip.clone(), img_size)
    gc.remove()
    return cam


# ============================================================
# Attention Map Extraction
# ============================================================
def extract_sqa_attention(model, clip, img_size=IMG_SIZE):
    """Extract cross-attention weights from SQA (CrossAttention) adapter."""
    attn_weights = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple) and len(output) == 2:
            attn_weights['cross'] = output[1].detach().cpu()

    cross_attn = model.net3D.adapter.cross_attn
    handle = cross_attn.register_forward_hook(hook_fn)

    # Patch forward to get attention weights
    old_forward = cross_attn.forward

    def patched_forward(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False
        return old_forward(*args, **kwargs)

    cross_attn.forward = patched_forward

    with torch.no_grad():
        model(clip)

    cross_attn.forward = old_forward
    handle.remove()

    if 'cross' in attn_weights:
        attn = attn_weights['cross'][0]  # [heads, 49, N_tokens]
        attn_avg = attn.mean(dim=0)  # [49, N]

        N = attn_avg.shape[1]
        T_tokens = 8
        H_tokens = W_tokens = 14
        if N == T_tokens * H_tokens * W_tokens:
            attn_spatial = attn_avg.view(49, T_tokens, H_tokens, W_tokens)
            attn_spatial = attn_spatial.mean(dim=1)  # [49, 14, 14]
            attn_spatial_sum = attn_spatial.sum(dim=0)  # [14, 14]
        else:
            query_importance = attn_avg.sum(dim=1)  # [49]
            attn_spatial_sum = query_importance.view(7, 7).numpy()

        if isinstance(attn_spatial_sum, torch.Tensor):
            attn_spatial_sum = attn_spatial_sum.numpy()

        if attn_spatial_sum.max() > 0:
            attn_spatial_sum = (attn_spatial_sum - attn_spatial_sum.min()) / \
                               (attn_spatial_sum.max() - attn_spatial_sum.min())

        attn_resized = F.interpolate(
            torch.tensor(attn_spatial_sum).unsqueeze(0).unsqueeze(0).float(),
            size=(img_size, img_size), mode='bilinear', align_corners=False
        ).squeeze().numpy()

        return attn_resized

    return None


def extract_bstf_attention(model, clip, img_size=IMG_SIZE):
    """Extract bidirectional cross-attention from BSTF (CrossAttention) fusion."""
    attn_maps = {}

    # Hook into the first scale's cross-attention blocks
    handles = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                attn_maps[name] = output[1].detach().cpu()
        return hook_fn

    # Find cross-attention layers in fusion
    if hasattr(model.fusion, 'cls_blocks'):
        for i, block in enumerate(model.fusion.cls_blocks):
            ca_block = block.cross_attention
            # Patch the cross_attn_2d_to_3d to return attention
            attn_2d_to_3d = ca_block.cross_attn_2d_to_3d

            old_fwd = attn_2d_to_3d.forward
            def make_patched(orig_fwd, hook_name):
                def patched(*args, **kwargs):
                    kwargs['return_attention'] = True
                    return orig_fwd(*args, **kwargs)
                return patched

            attn_2d_to_3d.forward = make_patched(old_fwd, f'2d_to_3d_scale{i}')
            handles.append((attn_2d_to_3d, old_fwd))

            h = attn_2d_to_3d.register_forward_hook(make_hook(f'2d_to_3d_scale{i}'))
            handles.append(h)

    with torch.no_grad():
        model(clip)

    # Restore
    for item in handles:
        if isinstance(item, tuple):
            module, old_fwd = item
            module.forward = old_fwd
        else:
            item.remove()

    if attn_maps:
        # Use first scale's 2D->3D attention
        key = list(attn_maps.keys())[0]
        attn = attn_maps[key][0]  # [H, N_q, N_kv]
        attn_avg = attn.mean(dim=0)  # [N_q, N_kv]

        # N_q = H*W of 2D feature, try to reshape
        n_q = attn_avg.shape[0]
        h = w = int(n_q ** 0.5)
        if h * w == n_q:
            # Sum over key dimension to get per-query importance
            query_attn = attn_avg.sum(dim=1)  # [N_q]
            attn_map = query_attn.view(h, w).numpy()
        else:
            attn_map = attn_avg.sum(dim=1).numpy()
            h = w = int(len(attn_map) ** 0.5)
            if h * w == len(attn_map):
                attn_map = attn_map.reshape(h, w)
            else:
                attn_map = np.zeros((7, 7))

        if attn_map.max() > 0:
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        attn_resized = F.interpolate(
            torch.tensor(attn_map).unsqueeze(0).unsqueeze(0).float(),
            size=(img_size, img_size), mode='bilinear', align_corners=False
        ).squeeze().numpy()

        return attn_resized

    return None


def run_attention(model, clip, method, fusion_type, img_size=IMG_SIZE):
    """Extract attention maps. Returns dict of name -> heatmap."""
    results = {}

    # SQA adapter attention
    if method == 'cross':
        print("  Extracting SQA cross-attention map...")
        sqa_attn = extract_sqa_attention(model, clip.clone(), img_size)
        if sqa_attn is not None:
            results['SQA Attention'] = sqa_attn
        else:
            print("  WARNING: Could not extract SQA attention")

    # BSTF fusion attention
    if fusion_type == 'CrossAttention':
        print("  Extracting BSTF cross-attention map...")
        bstf_attn = extract_bstf_attention(model, clip.clone(), img_size)
        if bstf_attn is not None:
            results['BSTF Attention'] = bstf_attn
        else:
            print("  WARNING: Could not extract BSTF attention")

    if not results:
        print(f"  INFO: No attention maps available for method='{method}', fusion='{fusion_type}'")
        print(f"        SQA attention requires method='cross', BSTF requires fusion='CrossAttention'")

    return results


# ============================================================
# BBox Detection
# ============================================================
def detect_bboxes(model, clip, conf_thresh=0.3, nms_thresh=0.5, img_size=IMG_SIZE):
    """Run inference and return detected bounding boxes.

    Returns:
        list of dicts: [{'bbox': [x1,y1,x2,y2], 'score': float, 'class_id': int, 'label': str}]
    """
    model.eval()
    with torch.no_grad():
        output = model(clip)

    # Inference output: [B, 4+num_classes, num_anchors]
    if isinstance(output, (list, tuple)):
        # Training mode output - skip bbox (need eval mode)
        print("  WARNING: Model returned training-mode output. Switching to eval mode...")
        model.eval()
        with torch.no_grad():
            output = model(clip)

    if output.dim() != 3:
        print(f"  WARNING: Unexpected output shape: {output.shape}")
        return []

    # output: [B, 4+num_classes, num_anchors]
    num_cls = model.detection_head.nc
    output = output[0]  # [4+num_classes, num_anchors]

    boxes = output[:4, :].T  # [num_anchors, 4] = cx, cy, w, h (scaled)
    scores = output[4:, :].T  # [num_anchors, num_classes]

    # Convert cx, cy, w, h -> x1, y1, x2, y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # [num_anchors, 4]

    # Normalize to [0, 1] then scale to img_size
    boxes_xyxy = boxes_xyxy / img_size
    boxes_xyxy = boxes_xyxy.clamp(0, 1) * img_size

    # Get max class score per anchor
    max_scores, class_ids = scores.max(dim=1)

    # Filter by confidence
    mask = max_scores > conf_thresh
    if mask.sum() == 0:
        # Try lower threshold
        if max_scores.max() > 0:
            print(f"  No detections above {conf_thresh}. Max score: {max_scores.max():.4f}")
        return []

    filtered_boxes = boxes_xyxy[mask]
    filtered_scores = max_scores[mask]
    filtered_classes = class_ids[mask]

    # NMS
    try:
        from torchvision.ops import nms
        keep = nms(filtered_boxes, filtered_scores, nms_thresh)
    except ImportError:
        # Simple NMS fallback
        keep = torch.arange(len(filtered_boxes))

    # Get labels
    dataset = 'ava' if num_cls == 80 else 'ucf'
    labels = AVA_LABELS if dataset == 'ava' else UCF_LABELS

    detections = []
    for idx in keep[:20]:  # Limit to top 20
        cls_id = filtered_classes[idx].item()
        label = labels[cls_id] if cls_id < len(labels) else f"class_{cls_id}"
        # Skip unused AVA classes (empty label)
        if not label:
            continue
        detections.append({
            'bbox': filtered_boxes[idx].cpu().numpy().tolist(),
            'score': filtered_scores[idx].item(),
            'class_id': cls_id,
            'label': label,
        })

    return detections


# ============================================================
# Overlay / Drawing
# ============================================================
def overlay_heatmap(image, heatmap, alpha=0.5, colormap='jet'):
    """Overlay heatmap on PIL image. Returns numpy array [H, W, 3] in [0, 1]."""
    img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    cmap = plt.colormaps[colormap]
    heatmap_colored = cmap(heatmap)[:, :, :3]
    overlay = (1 - alpha) * img_array + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def draw_bboxes(image, detections, img_size=IMG_SIZE):
    """Draw bounding boxes on PIL image. Returns PIL Image."""
    img = image.resize((img_size, img_size)).copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, det in enumerate(detections):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['label']} {det['score']:.2f}"

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label background
        bbox = font.getbbox(label)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), label, fill="white", font=font)

    return img


# ============================================================
# Output / Display
# ============================================================
def save_panel(img_data, filepath, dpi=300):
    """Save a single visualization panel as PNG."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    if isinstance(img_data, Image.Image):
        ax.imshow(img_data)
    else:
        ax.imshow(img_data)
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(str(filepath), dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    plt.close(fig)


def show_results(results, title="YOWOFormer Visualization"):
    """Display all results in a matplotlib grid."""
    n = len(results)
    if n == 0:
        print("No results to display.")
        return

    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (name, img_data) in enumerate(results.items()):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        if isinstance(img_data, Image.Image):
            ax.imshow(img_data)
        else:
            ax.imshow(img_data)
        ax.set_title(name, fontsize=10)
        ax.axis('off')

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')

    fig.tight_layout()
    plt.show()


def save_results(results, output_dir, prefix="viz"):
    """Save all results as individual PNG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, img_data in results.items():
        safe_name = re.sub(r'[^\w\-]', '_', name.lower())
        filepath = output_dir / f"{prefix}_{safe_name}.png"
        save_panel(img_data, filepath)
        print(f"  Saved: {filepath}")


# ============================================================
# Interactive Mode
# ============================================================
def interactive_menu(args):
    """Fill in missing args via interactive prompts."""
    print("\n" + "=" * 50)
    print("  YOWOFormer Visualization Tool (Interactive)")
    print("=" * 50)

    # Checkpoint
    if not args.checkpoint:
        default_ckpt = DEFAULT_CHECKPOINT
        custom = input(f"\nCheckpoint path [{default_ckpt}]: ").strip()
        args.checkpoint = custom if custom else default_ckpt

    # Input type
    if not args.input_type:
        print("\nInput type:")
        print("  1) ava  - AVA Dataset frames")
        print("  2) video - Video file (.mp4/.avi)")
        choice = input("Select [1/2]: ").strip()
        args.input_type = 'video' if choice == '2' else 'ava'

    # AVA options
    if args.input_type == 'ava':
        if not args.video_id:
            args.video_id = input("Video ID (e.g., 9F2voT6QWvQ): ").strip()
        if not args.timestamp:
            ts = input("Timestamp (e.g., 940): ").strip()
            args.timestamp = int(ts) if ts else 940
        if not args.ava_frames:
            custom = input(f"AVA frames directory [{AVA_FRAMES_DEFAULT}]: ").strip()
            args.ava_frames = custom if custom else AVA_FRAMES_DEFAULT

    # Video options
    elif args.input_type == 'video':
        if not args.video:
            args.video = input("Video file path: ").strip()
        if args.frame is None:
            f = input("Frame index (default: last frame): ").strip()
            args.frame = int(f) if f else -1

    # Mode
    if not args.mode:
        print("\nVisualization modes:")
        print("  1) gradcam   - GradCAM heatmap")
        print("  2) attention - Attention maps (SQA/BSTF)")
        print("  3) bbox      - Bounding box detection")
        print("  4) all       - All of the above")
        choice = input("Select modes (comma-separated, e.g., 1,2 or 4): ").strip()
        mode_map = {'1': 'gradcam', '2': 'attention', '3': 'bbox', '4': 'all'}
        if '4' in choice or 'all' in choice:
            args.mode = ['all']
        else:
            args.mode = [mode_map.get(c.strip(), c.strip()) for c in choice.split(',')]

    # GradCAM target
    if 'gradcam' in args.mode or 'all' in args.mode:
        if not args.target_layer:
            print("\nGradCAM target layer:")
            print("  1) adapter   - 3D adapter output")
            print("  2) fusion    - Fusion cls output (multi-scale)")
            print("  3) detection - Detection head cls output")
            choice = input("Select [1/2/3, default=1]: ").strip()
            target_map = {'1': 'adapter', '2': 'fusion', '3': 'detection'}
            args.target_layer = target_map.get(choice, 'adapter')

    # Duration (clip mode)
    if not hasattr(args, 'duration') or args.duration == 0:
        dur = input("\nDuration in seconds (0 = single frame, e.g., 3): ").strip()
        args.duration = float(dur) if dur else 0
        if args.duration > 0:
            fps_input = input("Output video FPS [5]: ").strip()
            args.fps = int(fps_input) if fps_input else 5

    # Output
    if not args.save and not args.show:
        print("\nOutput options:")
        print("  1) show - Display in window")
        print("  2) save - Save to directory")
        print("  3) both - Show and save")
        choice = input("Select [1/2/3, default=3]: ").strip()
        if choice == '1':
            args.show = True
        elif choice == '2':
            args.save = input("Output directory [output/viz]: ").strip() or "output/viz"
        else:
            args.show = True
            args.save = input("Output directory [output/viz]: ").strip() or "output/viz"

    return args


# ============================================================
# Main Pipeline
# ============================================================
def run_visualization(args):
    """Main visualization pipeline."""
    modes = set(args.mode)
    if 'all' in modes:
        modes = {'gradcam', 'attention', 'bbox'}

    print(f"\nDevice: {DEVICE}")
    print(f"Modes: {', '.join(sorted(modes))}")

    # --- Load Model ---
    print(f"\nLoading model from: {args.checkpoint}")
    model, info = load_model(args.checkpoint, DEVICE)
    print(f"  Method: {info['method']}, Fusion: {info['fusion']}")
    print(f"  Dataset: {info['dataset']}, Classes: {info['num_classes']}")
    print(f"  Clip length: {info['clip_length']}")

    # --- Load Input ---
    print("\nLoading input...")
    if args.input_type == 'ava':
        clip, key_frame = load_clip_ava(
            args.video_id, args.timestamp,
            args.ava_frames or AVA_FRAMES_DEFAULT,
            info['clip_length'], IMG_SIZE
        )
        print(f"  AVA: {args.video_id} @ t={args.timestamp}")
    else:
        frame_idx = args.frame
        if frame_idx == -1:
            # Use last frame
            import cv2
            cap = cv2.VideoCapture(str(args.video))
            frame_idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            cap.release()
        clip, key_frame = load_clip_video(
            args.video, frame_idx,
            info['clip_length'], IMG_SIZE
        )
        print(f"  Video: {args.video} @ frame={frame_idx}")

    clip = clip.to(DEVICE)

    # --- Collect Results ---
    results = {}

    # Always include key frame
    key_img = np.array(key_frame.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    results['Key Frame'] = key_img

    # --- GradCAM ---
    if 'gradcam' in modes:
        target = args.target_layer or 'adapter'
        print(f"\nRunning GradCAM (target: {target})...")
        cam = run_gradcam(model, clip, info['method'], target, IMG_SIZE)
        overlay = overlay_heatmap(key_frame, cam, args.alpha, args.colormap)
        results[f'GradCAM ({target})'] = overlay
        print(f"  GradCAM max={cam.max():.3f}")

    # --- Attention Maps ---
    if 'attention' in modes:
        print("\nExtracting attention maps...")
        attn_maps = run_attention(model, clip, info['method'], info['fusion'], IMG_SIZE)
        for name, attn in attn_maps.items():
            cmap_name = 'viridis' if 'SQA' in name else 'plasma'
            overlay = overlay_heatmap(key_frame, attn, args.alpha, cmap_name)
            results[name] = overlay
            print(f"  {name} max={attn.max():.3f}")

    # --- BBox Detection ---
    if 'bbox' in modes:
        print(f"\nRunning detection (conf>{args.conf_thresh})...")
        detections = detect_bboxes(model, clip, args.conf_thresh, 0.5, IMG_SIZE)
        print(f"  Found {len(detections)} detections")
        for det in detections[:5]:
            print(f"    {det['label']}: {det['score']:.3f}")

        if detections:
            bbox_img = draw_bboxes(key_frame, detections, IMG_SIZE)
            results['Detection'] = bbox_img

            # Also overlay bbox on GradCAM if available
            if 'gradcam' in modes:
                cam = run_gradcam(model, clip, info['method'], args.target_layer or 'adapter', IMG_SIZE)
                overlay = overlay_heatmap(key_frame, cam, args.alpha, args.colormap)
                overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
                bbox_gradcam = draw_bboxes(overlay_pil, detections, IMG_SIZE)
                results['GradCAM + BBox'] = bbox_gradcam

    # --- Output ---
    print(f"\nResults: {len(results)} panels")

    if args.save:
        print(f"Saving to: {args.save}")
        prefix = f"{args.video_id}_{args.timestamp}" if args.input_type == 'ava' else "video"
        save_results(results, args.save, prefix)

    if args.show:
        matplotlib.use('TkAgg')  # Switch to interactive backend
        title = f"YOWOFormer - {info['method'].upper()} + {info['fusion']}"
        show_results(results, title)

    print("\nDone!")
    return results


# ============================================================
# Demo Mode (3-panel for GitHub)
# ============================================================
def render_demo_frame(model, clip, key_frame, info, args):
    """Render 3 demo panels: Detection | SQA GradCAM+BBox | Fusion GradCAM+BBox."""
    panels = {}
    conf = args.conf_thresh

    # --- Detection (BBox + action labels) ---
    detections = detect_bboxes(model, clip, conf, 0.5, IMG_SIZE)

    bbox_img = draw_bboxes(key_frame, detections, IMG_SIZE)
    panels['Detection'] = np.array(bbox_img).astype(np.float32) / 255.0

    # --- GradCAM SQA (adapter) + BBox ---
    cam_sqa = run_gradcam(model, clip, info['method'], 'adapter', IMG_SIZE)
    overlay_sqa = overlay_heatmap(key_frame, cam_sqa, args.alpha, 'jet')
    overlay_sqa_pil = Image.fromarray((overlay_sqa * 255).astype(np.uint8))
    sqa_bbox = draw_bboxes(overlay_sqa_pil, detections, IMG_SIZE)
    panels['SQA GradCAM'] = np.array(sqa_bbox).astype(np.float32) / 255.0

    # --- GradCAM Fusion (BSTF/Simple) + BBox ---
    cam_fusion = run_gradcam(model, clip, info['method'], 'fusion', IMG_SIZE)
    overlay_fusion = overlay_heatmap(key_frame, cam_fusion, args.alpha, 'jet')
    overlay_fusion_pil = Image.fromarray((overlay_fusion * 255).astype(np.uint8))
    fusion_name = info['fusion']
    fusion_bbox = draw_bboxes(overlay_fusion_pil, detections, IMG_SIZE)
    panels[f'{fusion_name} GradCAM'] = np.array(fusion_bbox).astype(np.float32) / 255.0

    return panels


def run_demo_clip(args):
    """Render 3-panel demo video for GitHub."""
    import cv2

    duration = args.duration if args.duration > 0 else 3.0
    fps = args.fps

    print(f"\n{'='*50}")
    print(f"  Demo Mode (3-panel for GitHub)")
    print(f"{'='*50}")
    print(f"Device: {DEVICE}")
    print(f"Duration: {duration}s, FPS: {fps}")

    # --- Load Model ---
    print(f"\nLoading model: {args.checkpoint}")
    model, info = load_model(args.checkpoint, DEVICE)
    print(f"  {info['backbone2D']} + VideoMAE-{info['videomae_variant']}")
    print(f"  Adapter: {info['method'].upper()}, Fusion: {info['fusion']}")

    # --- Frame indices ---
    if args.input_type == 'ava':
        frames_root = args.ava_frames or AVA_FRAMES_DEFAULT
        start_ts = args.timestamp
        key_frame_idx_base = (start_ts - 900) * 30 + 1
        total_source_frames = int(duration * 30)
        step = max(1, 30 // fps)
        frame_offsets = list(range(0, total_source_frames, step))
        print(f"  AVA: {args.video_id} t={start_ts}, {len(frame_offsets)} frames")
    else:
        cap = cv2.VideoCapture(str(args.video))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        start_frame = args.frame if args.frame and args.frame >= 0 else 0
        total_source_frames = int(duration * video_fps)
        step = max(1, int(video_fps / fps))
        frame_offsets = list(range(0, total_source_frames, step))
        print(f"  Video: {args.video} frame={start_frame}, {len(frame_offsets)} frames")

    # --- Process first frame to get grid size ---
    print("\nRendering...")
    if args.input_type == 'ava':
        clip, key_frame = _load_clip_ava_by_frame(
            args.video_id, key_frame_idx_base, frames_root, info['clip_length'], IMG_SIZE
        )
    else:
        clip, key_frame = load_clip_video(args.video, start_frame, info['clip_length'], IMG_SIZE)

    clip = clip.to(DEVICE)
    panels = render_demo_frame(model, clip, key_frame, info, args)
    grid = compose_grid(panels, IMG_SIZE)
    grid_h, grid_w = grid.shape[:2]

    # --- Video writer ---
    output_dir = Path(args.save or 'output/demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_type == 'ava':
        vid_name = f"demo_{args.video_id}_t{start_ts}_{duration}s"
    else:
        vid_name = f"demo_{Path(args.video).stem}_{duration}s"

    video_path = output_dir / f"{vid_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (grid_w, grid_h))
    writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # --- Remaining frames ---
    for fi, offset in enumerate(frame_offsets[1:], 1):
        if args.input_type == 'ava':
            fidx = key_frame_idx_base + offset
            clip, key_frame = _load_clip_ava_by_frame(
                args.video_id, fidx, frames_root, info['clip_length'], IMG_SIZE
            )
        else:
            clip, key_frame = load_clip_video(
                args.video, start_frame + offset, info['clip_length'], IMG_SIZE
            )

        clip = clip.to(DEVICE)
        panels = render_demo_frame(model, clip, key_frame, info, args)
        grid = compose_grid(panels, IMG_SIZE)
        writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        print(f"  [{fi+1}/{len(frame_offsets)}] offset={offset}", end='\r')

    writer.release()

    # --- Also save as GIF for GitHub ---
    gif_path = output_dir / f"{vid_name}.gif"
    print(f"\n\nConverting to GIF...")
    try:
        # Re-read video and convert to GIF using PIL
        cap = cv2.VideoCapture(str(video_path))
        gif_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gif_frames.append(Image.fromarray(frame_rgb))
        cap.release()

        if gif_frames:
            gif_frames[0].save(
                str(gif_path), save_all=True, append_images=gif_frames[1:],
                duration=int(1000 / fps), loop=0, optimize=True
            )
            print(f"  GIF saved: {gif_path}")
    except Exception as e:
        print(f"  GIF conversion failed: {e}")

    print(f"\nVideo saved: {video_path}")
    print(f"  {grid_w}x{grid_h}, {fps}fps, {len(frame_offsets)} frames")
    print(f"\nFor GitHub README:")
    print(f"  ![YOWOFormer Demo]({gif_path.name})")
    print("Done!")


# ============================================================
# Video Clip Rendering
# ============================================================
def render_single_frame(model, clip, key_frame, info, modes, args):
    """Render all visualization panels for a single frame. Returns dict of name -> numpy [H,W,3]."""
    panels = {}

    key_img = np.array(key_frame.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    panels['Key Frame'] = key_img

    if 'gradcam' in modes:
        target = args.target_layer or 'adapter'
        cam = run_gradcam(model, clip, info['method'], target, IMG_SIZE)
        panels[f'GradCAM'] = overlay_heatmap(key_frame, cam, args.alpha, args.colormap)

    if 'attention' in modes:
        attn_maps = run_attention(model, clip, info['method'], info['fusion'], IMG_SIZE)
        for name, attn in attn_maps.items():
            cmap_name = 'viridis' if 'SQA' in name else 'plasma'
            panels[name] = overlay_heatmap(key_frame, attn, args.alpha, cmap_name)

    if 'bbox' in modes:
        detections = detect_bboxes(model, clip, args.conf_thresh, 0.5, IMG_SIZE)
        if detections:
            bbox_img = draw_bboxes(key_frame, detections, IMG_SIZE)
            panels['Detection'] = np.array(bbox_img).astype(np.float32) / 255.0

    return panels


def compose_grid(panels, img_size=IMG_SIZE):
    """Compose multiple panels into a single grid image. Returns numpy [H, W, 3] uint8."""
    n = len(panels)
    if n == 0:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    pad = 4  # padding between panels
    label_h = 20  # height for label text
    cell_w = img_size + pad
    cell_h = img_size + label_h + pad
    grid_w = cols * cell_w + pad
    grid_h = rows * cell_h + pad

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    for idx, (name, img_data) in enumerate(panels.items()):
        r, c = divmod(idx, cols)
        x = c * cell_w + pad
        y = r * cell_h + pad + label_h

        # Convert to uint8
        if isinstance(img_data, Image.Image):
            frame = np.array(img_data.resize((img_size, img_size)))
        elif img_data.max() <= 1.0:
            frame = (img_data * 255).astype(np.uint8)
        else:
            frame = img_data.astype(np.uint8)

        if frame.shape[:2] != (img_size, img_size):
            from PIL import Image as PILImage
            frame = np.array(PILImage.fromarray(frame).resize((img_size, img_size)))

        grid[y:y + img_size, x:x + img_size] = frame[:, :, :3]

        # Draw label using PIL
        label_img = Image.fromarray(grid[y - label_h:y, x:x + img_size])
        draw = ImageDraw.Draw(label_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((4, 2), name, fill=(0, 0, 0), font=font)
        grid[y - label_h:y, x:x + img_size] = np.array(label_img)

    return grid


def run_visualization_clip(args):
    """Render visualization over multiple frames and save as video."""
    import cv2

    modes = set(args.mode)
    if 'all' in modes:
        modes = {'gradcam', 'attention', 'bbox'}

    duration = args.duration
    fps = args.fps

    print(f"\nDevice: {DEVICE}")
    print(f"Modes: {', '.join(sorted(modes))}")
    print(f"Duration: {duration}s, Output FPS: {fps}")

    # --- Load Model ---
    print(f"\nLoading model from: {args.checkpoint}")
    model, info = load_model(args.checkpoint, DEVICE)
    print(f"  Method: {info['method']}, Fusion: {info['fusion']}")
    print(f"  Clip length: {info['clip_length']}")

    # --- Determine frames to process ---
    if args.input_type == 'ava':
        frames_root = args.ava_frames or AVA_FRAMES_DEFAULT
        start_ts = args.timestamp
        # AVA = 30fps, process at output fps rate
        total_source_frames = int(duration * 30)
        step = max(1, 30 // fps)  # e.g., fps=5 → step=6 (every 6th frame)
        frame_indices = list(range(0, total_source_frames, step))

        print(f"  AVA: {args.video_id}, t={start_ts}, {len(frame_indices)} frames (step={step})")

    else:  # video file
        cap = cv2.VideoCapture(str(args.video))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        start_frame = args.frame if args.frame and args.frame >= 0 else 0
        total_source_frames = int(duration * video_fps)
        step = max(1, int(video_fps / fps))
        frame_indices = list(range(0, total_source_frames, step))

        print(f"  Video: {args.video}, start={start_frame}, {len(frame_indices)} frames (step={step})")

    # --- Process first frame to get grid dimensions ---
    print("\nRendering frames...")

    if args.input_type == 'ava':
        key_frame_idx_base = (start_ts - 900) * 30 + 1
        first_fidx = key_frame_idx_base + frame_indices[0]
        # Compute corresponding timestamp for load_clip_ava
        first_ts = 900 + (first_fidx - 1) / 30
        clip, key_frame = load_clip_ava(
            args.video_id, start_ts, frames_root, info['clip_length'], IMG_SIZE
        )
    else:
        first_frame = start_frame + frame_indices[0]
        clip, key_frame = load_clip_video(args.video, first_frame, info['clip_length'], IMG_SIZE)

    clip = clip.to(DEVICE)
    panels = render_single_frame(model, clip, key_frame, info, modes, args)
    grid = compose_grid(panels, IMG_SIZE)
    grid_h, grid_w = grid.shape[:2]

    # --- Setup video writer ---
    output_dir = Path(args.save or 'output/viz')
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_type == 'ava':
        video_name = f"{args.video_id}_t{start_ts}_{duration}s"
    else:
        video_name = f"{Path(args.video).stem}_f{start_frame}_{duration}s"

    video_path = output_dir / f"{video_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (grid_w, grid_h))

    # Write first frame
    writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # --- Process remaining frames ---
    for fi, offset in enumerate(frame_indices[1:], 1):
        progress = f"[{fi + 1}/{len(frame_indices)}]"

        if args.input_type == 'ava':
            # Load clip centered on this frame
            fidx = key_frame_idx_base + offset
            # Reconstruct timestamp (approximate)
            ts_approx = 900 + (fidx - 1) / 30
            # Load the specific key frame and build clip around it
            clip, key_frame = _load_clip_ava_by_frame(
                args.video_id, fidx, frames_root, info['clip_length'], IMG_SIZE
            )
        else:
            frame_idx = start_frame + offset
            clip, key_frame = load_clip_video(args.video, frame_idx, info['clip_length'], IMG_SIZE)

        clip = clip.to(DEVICE)
        panels = render_single_frame(model, clip, key_frame, info, modes, args)
        grid = compose_grid(panels, IMG_SIZE)
        writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        print(f"  {progress} frame offset={offset}", end='\r')

    writer.release()
    print(f"\n\nVideo saved: {video_path}")
    print(f"  Resolution: {grid_w}x{grid_h}, FPS: {fps}, Frames: {len(frame_indices)}")
    print("Done!")


def _load_clip_ava_by_frame(video_id, key_fidx, frames_root, clip_length=16, img_size=224):
    """Load a clip from AVA dataset using a specific frame index as key frame."""
    frames = []
    for i in reversed(range(clip_length)):
        fidx = max(1, key_fidx - i)
        fpath = Path(frames_root) / video_id / f"{video_id}_{fidx:06d}.jpg"
        if not fpath.exists():
            fpath = Path(frames_root) / video_id / f"{video_id}_{key_fidx:06d}.jpg"
        if not fpath.exists():
            raise FileNotFoundError(f"Frame not found: {fpath}")
        img = Image.open(fpath).convert("RGB").resize((img_size, img_size))
        frames.append(img)

    key_frame = frames[-1].copy()
    clip = _frames_to_tensor(frames, img_size)
    return clip, key_frame


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="YOWOFormer Visualization Tool - GradCAM, Attention Map, BBox Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python visualize.py

  # GradCAM on AVA dataset
  python visualize.py --checkpoint weights/best.pth \\
    --input-type ava --video-id 9F2voT6QWvQ --timestamp 940 \\
    --mode gradcam --target-layer adapter --save output/

  # All visualizations on video file
  python visualize.py --checkpoint weights/best.pth \\
    --input-type video --video clip.mp4 --frame 100 \\
    --mode all --show --save output/

  # Attention maps only
  python visualize.py --checkpoint weights/best.pth \\
    --input-type ava --video-id VIDEO_ID --timestamp 940 \\
    --mode attention --show
        """)

    # Required
    parser.add_argument('--checkpoint', '-c', type=str, default=DEFAULT_CHECKPOINT,
                        help=f'Path to model checkpoint (.pth) (default: {DEFAULT_CHECKPOINT})')

    # Input
    parser.add_argument('--input-type', '-i', type=str, choices=['ava', 'video'], default=None,
                        help='Input type: ava (dataset frames) or video (video file)')
    parser.add_argument('--video-id', type=str, default=None,
                        help='AVA video ID (e.g., 9F2voT6QWvQ)')
    parser.add_argument('--timestamp', '-t', type=int, default=None,
                        help='AVA timestamp (e.g., 940)')
    parser.add_argument('--ava-frames', type=str, default=None,
                        help=f'AVA frames directory (default: {AVA_FRAMES_DEFAULT})')
    parser.add_argument('--video', '-v', type=str, default=None,
                        help='Path to video file (.mp4/.avi)')
    parser.add_argument('--frame', '-f', type=int, default=None,
                        help='Frame index for video file (default: last frame)')

    # Visualization modes
    parser.add_argument('--mode', '-m', nargs='+', default=None,
                        choices=['gradcam', 'attention', 'bbox', 'all'],
                        help='Visualization mode(s): gradcam, attention, bbox, all')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='Demo mode: 3-panel (Detection | SQA GradCAM+BBox | Fusion GradCAM+BBox) + GIF for GitHub')
    parser.add_argument('--target-layer', type=str, default=None,
                        choices=['adapter', 'fusion', 'detection'],
                        help='GradCAM target layer (default: adapter)')

    # Display options
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Heatmap overlay transparency (default: 0.5)')
    parser.add_argument('--colormap', type=str, default='jet',
                        help='Heatmap colormap: jet, viridis, hot, etc. (default: jet)')
    parser.add_argument('--conf-thresh', type=float, default=0.3,
                        help='Confidence threshold for bbox detection (default: 0.3)')

    # Clip / Video output
    parser.add_argument('--duration', type=float, default=0,
                        help='Duration in seconds for video clip output (e.g., 3). 0 = single frame')
    parser.add_argument('--fps', type=int, default=5,
                        help='Output video FPS (default: 5)')

    # Output
    parser.add_argument('--save', '-s', type=str, default=None,
                        help='Save output to directory')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Display results in matplotlib window')

    return parser.parse_args()


def main():
    args = parse_args()

    # Check if enough args for CLI mode, otherwise go interactive
    needs_interactive = (
        args.checkpoint is None or
        args.input_type is None or
        args.mode is None
    )

    if needs_interactive and not args.demo:
        args = interactive_menu(args)

    if args.demo:
        # Demo mode: set defaults if not provided
        if not args.mode:
            args.mode = ['all']
        run_demo_clip(args)
    elif args.duration > 0:
        run_visualization_clip(args)
    else:
        run_visualization(args)


if __name__ == "__main__":
    main()
