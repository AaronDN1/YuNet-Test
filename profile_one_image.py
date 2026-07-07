"""Print per-stage detection timings for one image.

Usage:
    python profile_one_image.py path/to/image.jpg
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ensemble_detector import ENABLE_YOLOX_VOTER, EnsembleFaceDetector
from image_io import load_image
from main import _find_second_model, _find_yolox_model, _find_yunet_model


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    source = Path(sys.argv[1]).resolve()
    if not source.is_file():
        print(f"File not found: {source}")
        return 1

    yunet_path = _find_yunet_model()
    second_path = _find_second_model()
    if yunet_path is None or second_path is None:
        print("Missing YuNet or CenterFace model in models/.")
        return 1

    yolox_path = _find_yolox_model() if ENABLE_YOLOX_VOTER else None
    detector = EnsembleFaceDetector(yunet_path, second_path, yolox_path)

    image = load_image(source)
    t0 = time.perf_counter()
    result = detector.detect_debug(image)
    total_ms = (time.perf_counter() - t0) * 1000.0

    stage = int(result["escalation_stage"])
    faces = len(result["faces"])
    print(f"image: {source.name} ({image.shape[1]}x{image.shape[0]})")
    print(f"yolox_enabled: {ENABLE_YOLOX_VOTER and yolox_path is not None}")
    print(f"escalation_stage: {stage}")
    print(f"faces_accepted: {faces}")
    print("stage_timings_ms:")
    for name, ms in result.get("stage_timings_ms", {}).items():
        print(f"  {name}: {ms:.1f} ms")
    print(f"total: {total_ms:.1f} ms")
    if faces:
        print(f"-> accepted at stage {stage} ({'fast exit' if stage == 0 else 'escalated'})")
    else:
        print("-> quarantine (no faces accepted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
