"""CenterFace detector (MIT) running through OpenCV's DNN module.

CenterFace is an anchor-free CenterNet-style face detector. Its weights are
MIT-licensed (Star-Clouds/CenterFace), so they are safe for commercial use and
can ship with this repository.

IMPORTANT: this model is exported with a fixed ONNX input shape. OpenCV's DNN
engine will run it at other sizes, but it retains internal shape state between
forward passes -- if you run one image at a large size and the next at a smaller
size, the second decode returns garbage. To stay correct across a batch of
mixed-size images we always run at a single FIXED letterboxed input size, so the
network shape never changes between calls.

As with the YuNet proposer, this detector returns candidates down to a low score
on purpose: precision comes later from cross-model agreement in the ensemble.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anonymizer import Box
from boxops import nms

SCORE_THRESHOLD = 0.35
NMS_THRESHOLD = 0.30
# Fixed square input (multiple of 32). Larger = better small-face recall, slower.
INPUT_SIZE = 768
STRIDE = 4
OUTPUT_NAMES = ("537", "538", "539", "540")  # heatmap, scale, offset, landmarks


class CenterFaceDetector:
    def __init__(
        self,
        model_path: Path,
        score_threshold: float = SCORE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        input_size: int = INPUT_SIZE,
        max_side: int | None = None,  # accepted for API compatibility; unused
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"CenterFace model not found: {model_path}")

        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = max(32, int(round(input_size / 32) * 32))

    def detect(self, image: np.ndarray) -> list[Box]:
        return self._detect_scaled(image, 0, 0)

    def detect_candidates(self, image: np.ndarray) -> list[Box]:
        height, width = image.shape[:2]
        boxes = self._detect_scaled(image, 0, 0)
        if max(width, height) >= self.input_size:
            boxes.extend(self.detect_tiles(image, rows=2, cols=2))
        return nms(boxes, self.nms_threshold)

    def detect_tiles(self, image: np.ndarray, rows: int, cols: int) -> list[Box]:
        height, width = image.shape[:2]
        tile_w = width / cols
        tile_h = height / rows
        overlap_x = tile_w * 0.20
        overlap_y = tile_h * 0.20
        boxes: list[Box] = []

        for row in range(rows):
            for col in range(cols):
                x1 = int(max(0, col * tile_w - overlap_x))
                y1 = int(max(0, row * tile_h - overlap_y))
                x2 = int(min(width, (col + 1) * tile_w + overlap_x))
                y2 = int(min(height, (row + 1) * tile_h + overlap_y))
                tile = image[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                boxes.extend(self._detect_scaled(tile, x1, y1))
        return boxes

    def _detect_scaled(self, image: np.ndarray, x_offset: int, y_offset: int) -> list[Box]:
        height, width = image.shape[:2]
        if width < 8 or height < 8:
            return []

        # Letterbox into a fixed square so the net input shape never changes.
        size = self.input_size
        ratio = min(size / height, size / width)
        new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, scalefactor=1.0, size=(size, size), mean=(0, 0, 0), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        heatmap, scale, offset, _ = self.net.forward(list(OUTPUT_NAMES))
        return self._decode(heatmap, scale, offset, ratio, new_w, new_h, x_offset, y_offset)

    def _decode(
        self,
        heatmap: np.ndarray,
        scale: np.ndarray,
        offset: np.ndarray,
        ratio: float,
        content_w: int,
        content_h: int,
        x_offset: int,
        y_offset: int,
    ) -> list[Box]:
        heat = heatmap[0, 0]
        scale_h_map, scale_w_map = scale[0, 0], scale[0, 1]
        offset_y_map, offset_x_map = offset[0, 0], offset[0, 1]

        rows, cols = np.where(heat > self.score_threshold)
        if rows.size == 0:
            return []

        scores = heat[rows, cols]

        # Guard against pathological inputs: cap candidate count and clamp the
        # exponent so exp() can't overflow. Real faces are unaffected.
        if rows.size > 2000:
            top = np.argsort(scores)[::-1][:2000]
            rows, cols, scores = rows[top], cols[top], scores[top]

        box_h = np.exp(np.clip(scale_h_map[rows, cols], None, 6.0)) * STRIDE
        box_w = np.exp(np.clip(scale_w_map[rows, cols], None, 6.0)) * STRIDE
        off_y = offset_y_map[rows, cols]
        off_x = offset_x_map[rows, cols]

        # Center in the fixed letterboxed frame.
        cx = (cols + off_x + 0.5) * STRIDE
        cy = (rows + off_y + 0.5) * STRIDE

        result: list[Box] = []
        for cxi, cyi, bw, bh, score in zip(cx, cy, box_w, box_h, scores):
            # Drop detections whose center falls in the padded (non-image) area.
            if cxi > content_w or cyi > content_h:
                continue
            w = bw / ratio
            h = bh / ratio
            if w <= 1 or h <= 1:
                continue
            x = (cxi - bw / 2) / ratio + x_offset
            y = (cyi - bh / 2) / ratio + y_offset
            result.append(Box(int(round(x)), int(round(y)), int(round(w)), int(round(h)), float(score)))
        return result
