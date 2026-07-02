from __future__ import annotations

import numpy as np

from anonymizer import Box


def iou(a: Box, b: Box) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def containment(a: Box, b: Box) -> float:
    """Fraction of the smaller box that lies inside the other box."""
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    inter_w = max(0, min(ax2, bx2) - max(a.x, b.x))
    inter_h = max(0, min(ay2, by2) - max(a.y, b.y))
    smaller_area = min(a.w * a.h, b.w * b.h)
    return (inter_w * inter_h) / smaller_area if smaller_area else 0.0


def same_face(a: Box, b: Box, iou_threshold: float) -> bool:
    if iou(a, b) > iou_threshold or containment(a, b) > 0.72:
        return True

    center_a = (a.x + a.w / 2.0, a.y + a.h / 2.0)
    center_b = (b.x + b.w / 2.0, b.y + b.h / 2.0)
    distance = float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))
    scale = max(a.w, a.h, b.w, b.h)
    area_ratio = max(a.w * a.h, b.w * b.h) / max(1, min(a.w * a.h, b.w * b.h))
    return distance < scale * 0.28 and area_ratio < 3.0


def nms(boxes: list[Box], threshold: float) -> list[Box]:
    """Greedy non-maximum suppression that keeps the highest-scoring box per cluster."""
    if not boxes:
        return []

    order = sorted(range(len(boxes)), key=lambda i: boxes[i].score, reverse=True)
    keep: list[Box] = []
    suppressed: set[int] = set()

    for i in order:
        if i in suppressed:
            continue
        current = boxes[i]
        keep.append(current)
        for j in order:
            if j == i or j in suppressed:
                continue
            if same_face(current, boxes[j], threshold):
                suppressed.add(j)
    return keep
