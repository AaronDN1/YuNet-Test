"""Download a SCRFD detector model into the local models folder.

This fetches InsightFace's ``buffalo_l`` bundle and extracts its SCRFD-10G
detector (``det_10g.onnx``) into ``models/``, where the app auto-detects it and
switches to the YuNet + SCRFD ensemble.

Licensing: InsightFace's hosted model weights are published for non-commercial
research use. The model architecture/code is MIT-licensed, but the weights carry
a separate license. Only use this file for commercial purposes if you have
confirmed the weights are licensed for that. This script is a convenience for
obtaining the file; it does not grant you any rights to the weights.

Usage:
    python get_scrfd_model.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MEMBER_SUFFIX = "det_10g.onnx"
OUTPUT_FILENAME = "det_10g.onnx"

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / OUTPUT_FILENAME

    if target.exists() and target.stat().st_size > 0:
        print(f"SCRFD model already present: {target}")
        return 0

    print("Licensing reminder: InsightFace's hosted weights are for non-commercial")
    print("research use unless you have separately cleared them. Continuing download.\n")

    print(f"Downloading: {BUFFALO_L_URL}")
    try:
        request = urllib.request.Request(BUFFALO_L_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (trusted URL)
            archive_bytes = response.read()
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        print(
            "You can download the zip manually in a browser, unzip it, and copy "
            f"'{MEMBER_SUFFIX}' into:\n  {MODELS_DIR}",
            file=sys.stderr,
        )
        return 1

    print(f"Downloaded {len(archive_bytes) / (1024 * 1024):.1f} MB. Extracting {MEMBER_SUFFIX}...")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = _find_member(archive, MEMBER_SUFFIX)
            if member is None:
                print(
                    f"ERROR: '{MEMBER_SUFFIX}' not found in the archive. Members:\n  "
                    + "\n  ".join(archive.namelist()),
                    file=sys.stderr,
                )
                return 1
            data = archive.read(member)
    except zipfile.BadZipFile as exc:
        print(f"ERROR: downloaded file is not a valid zip: {exc}", file=sys.stderr)
        return 1

    if len(data) < 1024:
        print("ERROR: extracted model is unexpectedly small; aborting.", file=sys.stderr)
        return 1

    target.write_bytes(data)
    print(f"Saved SCRFD model to: {target}")
    print("Restart the app. The log should now report the YuNet + SCRFD ensemble.")
    return 0


def _find_member(archive: zipfile.ZipFile, suffix: str) -> str | None:
    for name in archive.namelist():
        if name.replace("\\", "/").endswith(suffix):
            return name
    return None


if __name__ == "__main__":
    raise SystemExit(main())
