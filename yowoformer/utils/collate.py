"""
Custom collate function for handling variable-sized bounding boxes
"""

import torch


def collate_fn(batch):
    """
    Custom collate function for YOWOFormer dataloader
    Handles variable number of bounding boxes per sample

    Args:
        batch: List of tuples (clip, boxes, labels)

    Returns:
        clips: Tensor of video clips [B, C, T, H, W]
        boxes: List of tensors with variable sizes
        labels: List of tensors with variable sizes
    """
    clips = []
    boxes = []
    labels = []

    for sample in batch:
        if len(sample) == 3:
            clip, box, label = sample
            clips.append(clip)
            boxes.append(box)
            labels.append(label)
        else:
            # Handle different formats if needed
            clips.append(sample[0])
            if len(sample) > 1:
                boxes.append(sample[1] if sample[1] is not None else torch.empty(0, 4))
            if len(sample) > 2:
                labels.append(sample[2] if sample[2] is not None else torch.empty(0))

    # Stack clips (they should all be the same size)
    clips = torch.stack(clips, dim=0)

    # Keep boxes and labels as lists (don't stack them)
    # This allows variable number of boxes per image
    return clips, boxes, labels


def collate_fn_with_padding(batch):
    """
    Alternative collate function that pads boxes to same size

    Args:
        batch: List of tuples (clip, boxes, labels)

    Returns:
        clips: Tensor [B, C, T, H, W]
        boxes: Padded tensor [B, max_boxes, 4]
        labels: Padded tensor [B, max_boxes] or [B, max_boxes, num_classes]
        valid_mask: Boolean tensor indicating valid boxes [B, max_boxes]
    """
    clips = []
    boxes_list = []
    labels_list = []

    for sample in batch:
        clip, box, label = sample[:3]
        clips.append(clip)
        boxes_list.append(box)
        labels_list.append(label)

    # Stack clips
    clips = torch.stack(clips, dim=0)

    # Find max number of boxes
    max_boxes = max(len(b) for b in boxes_list)
    if max_boxes == 0:
        max_boxes = 1  # At least 1 for shape consistency

    # Pad boxes and labels
    batch_size = len(boxes_list)
    padded_boxes = torch.zeros(batch_size, max_boxes, 4)
    valid_mask = torch.zeros(batch_size, max_boxes, dtype=torch.bool)

    # Handle labels based on their shape
    if len(labels_list[0].shape) == 1:
        # Single label per box
        padded_labels = torch.zeros(batch_size, max_boxes, dtype=torch.long)
    else:
        # Multi-label per box
        num_classes = labels_list[0].shape[-1]
        padded_labels = torch.zeros(batch_size, max_boxes, num_classes)

    for i, (box, label) in enumerate(zip(boxes_list, labels_list)):
        num_boxes = len(box)
        if num_boxes > 0:
            padded_boxes[i, :num_boxes] = box
            padded_labels[i, :num_boxes] = label
            valid_mask[i, :num_boxes] = True

    return clips, padded_boxes, padded_labels, valid_mask