#!/usr/bin/env python3
"""
YOWOFormer Unified Evaluation Tool
====================================
Measures mAP (UCF101-24, AVA v2.2), GFLOPs, FPS, and Parameters.
Auto-detects model configuration from checkpoint.

Usage:
    # Everything
    python evaluate.py -c weights/best.pth --mode all

    # mAP only
    python evaluate.py -c weights/best.pth --mode map --dataset ucf
    python evaluate.py -c weights/best.pth --mode map --dataset ava

    # Stats only (fast, no dataset needed)
    python evaluate.py -c weights/best.pth --mode stats

    # Save results
    python evaluate.py -c weights/best.pth --mode all --save-json results.json
"""

import sys
import os
import csv
import time
import json
import heapq
import logging
import argparse
import pprint
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

from yowoformer.model.yowoformer import build_yowoformer
from yowoformer.utils.box import non_max_suppression, box_iou
from yowoformer.utils.collate import collate_fn


# ============================================================
# Model Loading (auto-detect from checkpoint)
# ============================================================
def load_model(checkpoint_path, device='cuda'):
    """Load YOWOFormer model from checkpoint. Auto-detect config."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})

    # Auto-detect videomae method
    method = cfg.get('videomae_method', None)
    if method is None:
        bb3d_cfg = cfg.get('BACKBONE3D', {})
        method = bb3d_cfg.get('METHOD', 'simple')

    clip_length = cfg.get('clip_length', 16)
    bb2d = cfg.get('backbone2D', 'yolov11_n')

    config = {
        'dataset': cfg.get('dataset', 'ava'),
        'data_root': cfg.get('data_root', 'data/AVA_Dataset'),
        'num_classes': cfg.get('num_classes', 80),
        'backbone2D': bb2d,
        'backbone3D': cfg.get('backbone3D', 'videomae'),
        'videomae_variant': cfg.get('videomae_variant', 'base'),
        'videomae_method': method,
        'fusion_module': cfg.get('fusion_module', 'Simple'),
        'mode': cfg.get('mode', 'decoupled'),
        'interchannels': cfg.get('interchannels', [256, 256, 256]),
        'img_size': cfg.get('img_size', 224),
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
        if 'YOLOv8' in config['BACKBONE2D']:
            config['BACKBONE2D']['YOLOv8']['PRETRAIN'] = {
                k: None for k in config['BACKBONE2D']['YOLOv8'].get('PRETRAIN', {})
            }

    model = build_yowoformer(config)

    # Load weights (support regular + EMA)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'ema_state_dict' in ckpt:
        state_dict = ckpt['ema_state_dict']
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [Model] Missing keys: {len(missing)}")
    if unexpected:
        print(f"  [Model] Unexpected keys: {len(unexpected)}")

    model.eval()
    model.to(device)

    info = {
        'backbone2D': bb2d,
        'backbone3D': cfg.get('backbone3D', 'videomae'),
        'videomae_variant': cfg.get('videomae_variant', 'base'),
        'videomae_method': method,
        'fusion_module': cfg.get('fusion_module', 'Simple'),
        'mode': cfg.get('mode', 'decoupled'),
        'dataset': cfg.get('dataset', 'ava'),
        'num_classes': cfg.get('num_classes', 80),
        'clip_length': clip_length,
        'img_size': cfg.get('img_size', 224),
        'epoch': ckpt.get('epoch', 'unknown'),
        'train_loss': ckpt.get('train_loss', None),
    }

    return model, config, info


# ============================================================
# UCF101-24 mAP Evaluation
# ============================================================
def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))
    ap = np.zeros((nc, tp.shape[1]))
    px = np.linspace(0, 1, 1000)

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]
        no = i.sum()
        if no == 0 or nl == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (nl + eps)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))
            ap[ci, j] = np.trapz(m_pre, m_rec)

    f1 = 2 * p * r / (p + r + eps)
    i = smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    ap50, ap_all = ap[:, 0], ap.mean(1)
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap_all.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


@torch.no_grad()
def evaluate_ucf(model, config, device, conf_thres=0.005, nms_thres=0.5,
                 iou_type='all', batch_size=16, num_workers=4):
    """Run UCF101-24 mAP evaluation."""
    from yowoformer.datasets.ucf.load_data import build_ucf_dataset

    print("\n" + "=" * 60)
    print("UCF101-24 mAP Evaluation")
    print("=" * 60)

    # Build dataset
    test_dataset = build_ucf_dataset(config, phase='test')
    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True, drop_last=False
    )
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Test batches: {len(test_loader):,}")

    # Setup IoU thresholds
    if iou_type == 'all':
        iou_thresholds = torch.linspace(0.5, 0.95, 10).to(device)
    else:
        iou_thresholds = torch.tensor([0.5]).to(device)
    n_iou = iou_thresholds.numel()

    img_size = config['img_size']
    metrics = []

    model.eval()
    pbar = tqdm(test_loader, desc='Evaluating UCF')
    for clips, boxes, labels in pbar:
        clips = clips.to(device)

        # Prepare targets: [batch_idx, class, x1, y1, x2, y2]
        targets = []
        for i, (bbox, label) in enumerate(zip(boxes, labels)):
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.to(device)
            else:
                bbox = torch.tensor(bbox).to(device)
            if isinstance(label, torch.Tensor):
                label = label.to(device)
            else:
                label = torch.tensor(label).to(device)

            if len(bbox) > 0:
                target = torch.zeros(bbox.shape[0], 6).to(device)
                target[:, 0] = i
                target[:, 1] = label if label.dim() == 1 else label.argmax(1)
                target[:, 2:] = bbox
                targets.append(target)

        if not targets:
            continue
        targets = torch.cat(targets, dim=0)

        # Inference + NMS
        outputs = model(clips)
        targets[:, 2:] *= torch.tensor([img_size] * 4, device=device).float()
        outputs = non_max_suppression(outputs, conf_threshold=conf_thres, iou_threshold=nms_thres)

        # Match detections to ground truth
        for i, output in enumerate(outputs):
            labels_i = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(
                output.shape[0] if output is not None else 0,
                n_iou, dtype=torch.bool
            ).to(device)

            if output is None or output.shape[0] == 0:
                if labels_i.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).to(device)))
                continue

            if labels_i.shape[0]:
                target_boxes = labels_i[:, 1:5].clone()
                target_classes = labels_i[:, 0].clone()
                iou = box_iou(target_boxes, output[:, :4])
                correct_class = target_classes.view(-1, 1) == output[:, 5].view(1, -1)

                for j, iou_thresh in enumerate(iou_thresholds):
                    x = torch.where((iou >= iou_thresh) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((
                            torch.stack(x, 1).float(),
                            iou[x[0], x[1]][:, None]
                        ), 1)
                        if x[0].shape[0] > 1:
                            matches = matches.cpu().numpy()
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        else:
                            matches = matches.cpu().numpy()
                        for match in matches:
                            pred_idx = int(match[1])
                            if pred_idx < correct.shape[0]:
                                correct[pred_idx, j] = True

            metrics.append((correct, output[:, 4], output[:, 5], labels_i[:, 0]))

    # Compute final metrics
    print("\nComputing mAP...")
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]

    if len(metrics) and metrics[0].any():
        tp, fp, precision, recall, map50, map_all = compute_ap(*metrics)
        print(f"\n  Precision:      {precision:.4f}")
        print(f"  Recall:         {recall:.4f}")
        print(f"  mAP@0.5:        {map50:.4f}")
        if iou_type == 'all':
            print(f"  mAP@0.5:0.95:   {map_all:.4f}")

        return {
            'dataset': 'UCF101-24',
            'precision': float(precision),
            'recall': float(recall),
            'mAP@0.5': float(map50),
            'mAP@0.5:0.95': float(map_all) if iou_type == 'all' else None,
        }
    else:
        print("  No valid detections or targets found!")
        return None


# ============================================================
# AVA v2.2 mAP Evaluation
# ============================================================
AVA_BLACK_LIST = [2, 16, 18, 19, 21, 23, 25, 31, 32, 33, 35, 39, 40, 42, 44, 50, 53, 55, 71, 75]


def ava_collate_fn(batch):
    """Custom collate for AVA test set (5-element tuples)."""
    clips, video_names, timestamps = [], [], []
    for b in batch:
        clips.append(b[0])
        if len(b) >= 5:
            video_names.append(b[3])
            timestamps.append(b[4])
        else:
            video_names.append("unknown")
            timestamps.append("0")
    clips = torch.stack(clips, dim=0)
    return clips, video_names, timestamps


def read_labelmap(labelmap_file):
    """Read AVA label map in pbtxt format."""
    labelmap = []
    class_ids = set()
    with open(labelmap_file, 'r') as f:
        name = ""
        for line in f:
            if line.startswith("  name:"):
                name = line.split('"')[1]
            elif line.startswith("  id:") or line.startswith("  label_id:"):
                class_id = int(line.strip().split(" ")[-1])
                labelmap.append({"id": class_id, "name": name})
                class_ids.add(class_id)
    return labelmap, class_ids


def read_ava_csv(csv_file, class_whitelist=None, capacity=0):
    """Read CSV file in AVA format with optional capacity limiting."""
    entries = defaultdict(list)
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    all_keys = set()

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) not in [2, 7, 8]:
                continue
            image_key = f"{row[0]},{float(row[1]):.6f}"
            all_keys.add(image_key)
            if len(row) == 2:
                continue

            x1, y1, x2, y2 = [float(n) for n in row[2:6]]
            action_id = int(row[6])
            if class_whitelist and action_id not in class_whitelist:
                continue

            score = 1.0
            if len(row) == 8:
                score = float(row[7])

            if capacity < 1 or len(entries[image_key]) < capacity:
                heapq.heappush(entries[image_key], (score, action_id, y1, x1, y2, x2))
            elif score > entries[image_key][0][0]:
                heapq.heapreplace(entries[image_key], (score, action_id, y1, x1, y2, x2))

    for image_key in entries:
        entry = sorted(entries[image_key], key=lambda x: -x[0])
        for item in entry:
            score, action_id, y1, x1, y2, x2 = item
            boxes[image_key].append([y1, x1, y2, x2])
            labels[image_key].append(action_id)
            scores[image_key].append(score)

    return boxes, labels, scores, all_keys


@torch.no_grad()
def evaluate_ava(model, config, device, conf_thres=0.01, nms_thres=0.5,
                 batch_size=32, num_workers=6):
    """Run AVA v2.2 mAP evaluation using official PascalDetectionEvaluator."""
    from yowoformer.datasets.ava.load_data import build_ava_dataset
    from yowoformer.evaluator.ava_official import object_detection_evaluation
    from yowoformer.evaluator.ava_official import standard_fields

    print("\n" + "=" * 60)
    print("AVA v2.2 mAP Evaluation")
    print("=" * 60)

    # File paths
    data_root = config.get('data_root', 'data/AVA_Dataset')
    labelmap_file = os.path.join(data_root, 'annotations', 'ava_v2.2',
                                  'ava_action_list_v2.2_for_activitynet_2019.pbtxt')
    groundtruth_file = os.path.join(data_root, 'annotations', 'ava_v2.2',
                                     'ava_val_v2.2.csv')

    if not os.path.exists(labelmap_file):
        print(f"  Label map not found: {labelmap_file}")
        return None
    if not os.path.exists(groundtruth_file):
        print(f"  Ground truth not found: {groundtruth_file}")
        return None

    # Build dataset
    test_dataset = build_ava_dataset(config, phase='test')
    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=ava_collate_fn, num_workers=num_workers,
        pin_memory=True, drop_last=False
    )
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Test batches: {len(test_loader):,}")

    img_size = config['img_size']

    # Step 1: Generate detections
    print("\nGenerating detections...")
    results = []
    model.eval()

    for clips, video_names, timestamps in tqdm(test_loader, desc='Processing AVA'):
        clips = clips.to(device)
        outputs = model(clips).cpu()

        for output, video_name, timestamp in zip(outputs, video_names, timestamps):
            output = output.unsqueeze(0)
            output = non_max_suppression(output, conf_thres, nms_thres)[0]

            if output is None or len(output) == 0:
                continue

            # Normalize to [0, 1]
            H = W = img_size
            output[:, 0] /= W
            output[:, 1] /= H
            output[:, 2] /= W
            output[:, 3] /= H

            for detection in output:
                x1, y1, x2, y2, score, class_idx = detection.tolist()
                class_id = int(class_idx) + 1  # 1-indexed for AVA

                if class_id in AVA_BLACK_LIST:
                    continue

                score = min(max(score, 0.0), 1.0)
                if score > conf_thres:
                    results.append([
                        video_name, float(timestamp),
                        round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3),
                        class_id, round(score, 3)
                    ])

    print(f"  Total detections: {len(results):,}")

    if not results:
        print("  No detections generated!")
        return None

    # Save temp detection CSV
    det_file = '/tmp/yowoformer_ava_detections.csv'
    with open(det_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)

    # Step 2: Official AVA evaluation
    print("\nRunning official AVA evaluation...")
    categories, class_whitelist = read_labelmap(labelmap_file)
    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(categories)

    # Load ground truth
    gt_boxes, gt_labels, _, included_keys = read_ava_csv(groundtruth_file, class_whitelist, capacity=0)
    for image_key in gt_boxes:
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key, {
                standard_fields.InputDataFields.groundtruth_boxes:
                    np.array(gt_boxes[image_key], dtype=float),
                standard_fields.InputDataFields.groundtruth_classes:
                    np.array(gt_labels[image_key], dtype=int),
                standard_fields.InputDataFields.groundtruth_difficult:
                    np.zeros(len(gt_boxes[image_key]), dtype=bool)
            })

    # Load detections
    det_boxes, det_labels, det_scores, _ = read_ava_csv(det_file, class_whitelist, capacity=50)
    for image_key in det_boxes:
        if image_key not in included_keys:
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key, {
                standard_fields.DetectionResultFields.detection_boxes:
                    np.array(det_boxes[image_key], dtype=float),
                standard_fields.DetectionResultFields.detection_classes:
                    np.array(det_labels[image_key], dtype=int),
                standard_fields.DetectionResultFields.detection_scores:
                    np.array(det_scores[image_key], dtype=float)
            })

    # Compute metrics
    print("Computing metrics...")
    metrics = pascal_evaluator.evaluate()

    map_05 = float(metrics.get('PascalBoxes_Precision/mAP@0.5IOU', 0))
    print(f"\n  mAP@0.5: {map_05:.4f}")

    # Per-category AP
    per_class = {}
    for category in categories:
        key = f"PascalBoxes_PerformanceByCategory/AP@0.5IOU/{category['name']}"
        if key in metrics:
            per_class[category['name']] = float(metrics[key])

    # Print top categories
    if per_class:
        sorted_cats = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top-10 categories:")
        for name, ap in sorted_cats[:10]:
            print(f"    {name:<25s} {ap:.4f}")

    # Cleanup temp file
    os.remove(det_file)

    return {
        'dataset': 'AVA v2.2',
        'mAP@0.5': map_05,
        'per_class_AP': per_class,
    }


# ============================================================
# Model Statistics (Parameters, GFLOPs, FPS)
# ============================================================
def count_parameters(model):
    """Count model parameters with breakdown."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        breakdown[name] = round(params / 1e6, 2)

    return {
        'total': total,
        'trainable': trainable,
        'total_M': round(total / 1e6, 2),
        'trainable_M': round(trainable / 1e6, 2),
        'breakdown': breakdown,
    }


def measure_gflops(model, input_shape, device):
    """Measure GFLOPs using fvcore -> ptflops -> thop fallback."""
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    # Try fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy_input)
            gflops = flops.total() / 1e9
        return round(gflops, 2), "fvcore"
    except Exception:
        pass

    # Try ptflops
    try:
        from ptflops import get_model_complexity_info
        def input_constructor(input_res):
            return torch.randn(1, *input_res).to(device)
        macs, _ = get_model_complexity_info(
            model, input_shape[1:],
            input_constructor=input_constructor,
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
        return round(macs / 1e9, 2), "ptflops"
    except Exception:
        pass

    # Try thop
    try:
        from thop import profile
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return round(macs / 1e9, 2), "thop"
    except Exception:
        pass

    return None, "failed"


def measure_fps(model, input_shape, device, warmup=10, runs=100):
    """Measure FPS with AMP autocast on CUDA."""
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    use_amp = device.type == 'cuda'

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.amp.autocast('cuda'):
                    _ = model(dummy_input)
                torch.cuda.synchronize()
            else:
                _ = model(dummy_input)

    # Measure
    timings = []
    with torch.no_grad():
        for _ in range(runs):
            if use_amp:
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.amp.autocast('cuda'):
                    _ = model(dummy_input)
                torch.cuda.synchronize()
            else:
                start = time.perf_counter()
                _ = model(dummy_input)
            timings.append(time.perf_counter() - start)

    timings = np.array(timings)
    fps = 1.0 / timings.mean()
    latency_ms = timings.mean() * 1000

    return {
        'fps': round(fps, 2),
        'latency_ms': round(latency_ms, 2),
        'fps_std': round(fps * timings.std() / timings.mean(), 2),
        'device': str(device),
        'amp': use_amp,
    }


def evaluate_stats(model, info, device, skip_gflops=False, skip_fps=False,
                   fps_warmup=10, fps_runs=100):
    """Measure all model statistics."""
    print("\n" + "=" * 60)
    print("Model Statistics")
    print("=" * 60)

    results = {}

    # Parameters
    params = count_parameters(model)
    results['parameters'] = params
    print(f"\n  Parameters:")
    print(f"    Total:     {params['total_M']:.2f}M")
    print(f"    Trainable: {params['trainable_M']:.2f}M")
    print(f"    Breakdown:")
    for name, count in params['breakdown'].items():
        print(f"      {name}: {count:.2f}M")

    clip_length = info.get('clip_length', 16)
    img_size = info.get('img_size', 224)
    input_shape = (1, 3, clip_length, img_size, img_size)

    # GFLOPs
    if not skip_gflops:
        print(f"\n  GFLOPs (input: {input_shape}):")
        gflops, method = measure_gflops(model, input_shape, device)
        if gflops:
            results['gflops'] = {'value': gflops, 'method': method}
            print(f"    GFLOPs: {gflops} ({method})")
        else:
            print(f"    GFLOPs: Failed to measure")

    # FPS
    if not skip_fps:
        print(f"\n  FPS (warmup={fps_warmup}, runs={fps_runs}):")
        fps_results = measure_fps(model, input_shape, device, fps_warmup, fps_runs)
        results['fps'] = fps_results
        print(f"    FPS:     {fps_results['fps']:.2f} +/- {fps_results['fps_std']:.2f}")
        print(f"    Latency: {fps_results['latency_ms']:.2f}ms")
        print(f"    AMP:     {fps_results['amp']}")

    return results


# ============================================================
# Main
# ============================================================
def print_summary(info, map_results, stats_results):
    """Print a nice summary table."""
    print("\n")
    print("+" + "=" * 50 + "+")
    print("|  YOWOFormer Evaluation Results" + " " * 20 + "|")
    print("+" + "=" * 50 + "+")

    # Model info
    print(f"|  Backbone 2D:  {info['backbone2D']:<34s}|")
    print(f"|  Backbone 3D:  VideoMAE-{info['videomae_variant']:<25s}|")
    print(f"|  Adapter:      {info['videomae_method']:<34s}|")
    print(f"|  Fusion:       {info['fusion_module']:<34s}|")
    print(f"|  Clip Length:  {info['clip_length']:<34d}|")
    print("+" + "-" * 50 + "+")

    # mAP results
    if map_results:
        ds = map_results.get('dataset', '')
        print(f"|  Dataset:      {ds:<34s}|")
        if 'mAP@0.5' in map_results:
            val = map_results['mAP@0.5']
            print(f"|  mAP@0.5:      {val:<34.4f}|")
        if map_results.get('mAP@0.5:0.95') is not None:
            val = map_results['mAP@0.5:0.95']
            print(f"|  mAP@0.5:0.95: {val:<34.4f}|")
        if 'precision' in map_results:
            print(f"|  Precision:    {map_results['precision']:<34.4f}|")
            print(f"|  Recall:       {map_results['recall']:<34.4f}|")
        print("+" + "-" * 50 + "+")

    # Stats results
    if stats_results:
        if 'parameters' in stats_results:
            p = stats_results['parameters']
            print(f"|  Params:       {p['total_M']:<34.2f}|")
        if 'gflops' in stats_results:
            g = stats_results['gflops']['value']
            print(f"|  GFLOPs:       {g:<34.2f}|")
        if 'fps' in stats_results:
            f = stats_results['fps']
            fps_str = f"{f['fps']:.1f} +/- {f['fps_std']:.1f}"
            print(f"|  FPS:          {fps_str:<34s}|")
            lat_str = f"{f['latency_ms']:.1f}ms"
            print(f"|  Latency:      {lat_str:<34s}|")

    print("+" + "=" * 50 + "+")


def main():
    parser = argparse.ArgumentParser(
        description='YOWOFormer Unified Evaluation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py -c weights/best.pth --mode all
  python evaluate.py -c weights/best.pth --mode map --dataset ucf
  python evaluate.py -c weights/best.pth --mode map --dataset ava
  python evaluate.py -c weights/best.pth --mode stats
  python evaluate.py -c weights/best.pth --mode stats --skip-gflops
  python evaluate.py -c weights/best.pth --mode all --save-json results.json
        """
    )

    # Required
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')

    # Mode
    parser.add_argument('--mode', type=str, default='all',
                        choices=['map', 'stats', 'all'],
                        help='Evaluation mode (default: all)')

    # Dataset (for mAP)
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['ucf', 'ava'],
                        help='Dataset for mAP eval (auto-detect from checkpoint if not set)')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override dataset root path')

    # Evaluation settings
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation (default: 16)')
    parser.add_argument('--conf-thres', type=float, default=None,
                        help='Confidence threshold (default: 0.005 for UCF, 0.01 for AVA)')
    parser.add_argument('--nms-thres', type=float, default=0.5,
                        help='NMS IoU threshold (default: 0.5)')
    parser.add_argument('--iou-type', type=str, default='all',
                        choices=['0.5', 'all'],
                        help='IoU threshold: 0.5 only or 0.5:0.95 (default: all)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Dataloader workers (default: 4)')

    # Stats settings
    parser.add_argument('--skip-gflops', action='store_true',
                        help='Skip GFLOPs measurement')
    parser.add_argument('--skip-fps', action='store_true',
                        help='Skip FPS measurement')
    parser.add_argument('--fps-warmup', type=int, default=10,
                        help='FPS warmup iterations (default: 10)')
    parser.add_argument('--fps-runs', type=int, default=100,
                        help='FPS measurement iterations (default: 100)')

    # Output
    parser.add_argument('--save-json', type=str, default=None,
                        help='Save results to JSON file')

    args = parser.parse_args()

    # Check checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path.name}")
    model, config, info = load_model(str(checkpoint_path), device)

    print(f"\nModel Configuration:")
    print(f"  Backbone 2D:  {info['backbone2D']}")
    print(f"  Backbone 3D:  VideoMAE-{info['videomae_variant']}")
    print(f"  Adapter:      {info['videomae_method']}")
    print(f"  Fusion:       {info['fusion_module']}")
    print(f"  Clip Length:  {info['clip_length']}")
    print(f"  Dataset:      {info['dataset']}")
    print(f"  Classes:      {info['num_classes']}")
    print(f"  Epoch:        {info['epoch']}")
    if info['train_loss'] is not None:
        print(f"  Train Loss:   {info['train_loss']:.4f}")

    # Determine dataset
    dataset = args.dataset or info['dataset']
    if args.data_root:
        config['data_root'] = args.data_root
    elif dataset == 'ucf' and config.get('data_root', '').startswith('data/AVA'):
        config['data_root'] = 'data/UCF101-24'
    elif dataset == 'ava' and config.get('data_root', '').startswith('data/UCF'):
        config['data_root'] = 'data/AVA_Dataset'

    # Default conf_thres per dataset
    if args.conf_thres is not None:
        conf_thres = args.conf_thres
    elif dataset == 'ava':
        conf_thres = 0.01
    else:
        conf_thres = 0.005

    # Run evaluations
    map_results = None
    stats_results = None

    if args.mode in ('map', 'all'):
        if dataset == 'ucf':
            map_results = evaluate_ucf(
                model, config, device,
                conf_thres=conf_thres,
                nms_thres=args.nms_thres,
                iou_type=args.iou_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        elif dataset == 'ava':
            map_results = evaluate_ava(
                model, config, device,
                conf_thres=conf_thres,
                nms_thres=args.nms_thres,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

    if args.mode in ('stats', 'all'):
        stats_results = evaluate_stats(
            model, info, device,
            skip_gflops=args.skip_gflops,
            skip_fps=args.skip_fps,
            fps_warmup=args.fps_warmup,
            fps_runs=args.fps_runs,
        )

    # Print summary
    print_summary(info, map_results, stats_results)

    # Save JSON
    if args.save_json:
        output = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint': str(checkpoint_path),
            'device': str(device),
            'model_info': info,
        }
        if map_results:
            output['map_results'] = map_results
        if stats_results:
            output['stats_results'] = stats_results
        with open(args.save_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.save_json}")


if __name__ == "__main__":
    main()
