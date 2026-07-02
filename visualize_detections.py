"""Visualize detector output to tune accuracy on real images.

Runs the ensemble over an input folder and writes annotated copies so you can
see, per image:

  - GREEN thick boxes:  faces the ensemble accepted (these get blurred).
  - BLUE thin boxes:     CenterFace proposals (with score).
  - RED thin boxes:      YuNet proposals (with score).
  - YELLOW thin boxes:   YOLOX-face proposals, if the model is present (with score).

Where a green box sits on overlapping blue+red boxes, the two models agreed.
Green boxes with no overlap were accepted on a single model's confidence.
Use this to spot misses (a face with no green box) and false positives (a green
box on a non-face), then adjust the thresholds in ensemble_detector.py.

Usage:
    python visualize_detections.py INPUT_DIR [OUTPUT_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from ensemble_detector import EnsembleFaceDetector
from image_io import iter_image_files, load_image
from main import _find_second_model, _find_yolox_model, _find_yunet_model


def _draw(image, boxes, color, thickness, label_scores):
    for box in boxes:
        cv2.rectangle(image, (box.x, box.y), (box.x + box.w, box.y + box.h), color, thickness)
        if label_scores:
            cv2.putText(
                image,
                f"{box.score:.2f}",
                (box.x, max(12, box.y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    input_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else input_dir.parent / f"{input_dir.name}_annotated"
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = EnsembleFaceDetector(_find_yunet_model(), _find_second_model(), _find_yolox_model())
    files = iter_image_files(input_dir, recursive=True)
    print(f"Annotating {len(files)} image(s) -> {output_dir}")

    summary: list[str] = []
    for index, source in enumerate(files, start=1):
        try:
            image = load_image(source)
            result = detector.detect_debug(image)
            canvas = image.copy()
            _draw(canvas, result["centerface"], (255, 0, 0), 1, True)    # blue
            _draw(canvas, result["yunet"], (0, 0, 255), 1, True)         # red
            _draw(canvas, result.get("yolox", []), (0, 220, 220), 1, True)  # yellow
            _draw(canvas, result["faces"], (0, 200, 0), 3, False)        # green

            destination = output_dir / source.relative_to(input_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination.with_suffix(".jpg")), canvas)

            line = (
                f"[{index}/{len(files)}] {source.name}: "
                f"accepted={len(result['faces'])} "
                f"centerface={len(result['centerface'])} yunet={len(result['yunet'])} "
                f"yolox={len(result.get('yolox', []))}"
            )
            print(line)
            summary.append(line)
        except Exception as exc:  # keep going on a bad file
            summary.append(f"[{index}/{len(files)}] {source.name}: ERROR {exc}")
            print(summary[-1])

    (output_dir / "_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"Done. Summary: {output_dir / '_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
