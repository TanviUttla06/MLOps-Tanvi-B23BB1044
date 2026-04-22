import torch
import numpy as np


def iou_score(pred, mask, num_classes=23):
    pred = torch.argmax(pred, dim=1)
    ious = []

    for cls in range(num_classes):
        inter = ((pred == cls) & (mask == cls)).sum().item()
        union = ((pred == cls) | (mask == cls)).sum().item()

        if union == 0:
            continue

        ious.append(inter / union)

    return np.mean(ious)


def dice_score(pred, mask, num_classes=23):
    pred = torch.argmax(pred, dim=1)
    dices = []

    for cls in range(num_classes):
        inter = ((pred == cls) & (mask == cls)).sum().item()
        total = (pred == cls).sum().item() + (mask == cls).sum().item()

        if total == 0:
            continue

        dices.append(2 * inter / total)

    return np.mean(dices)
