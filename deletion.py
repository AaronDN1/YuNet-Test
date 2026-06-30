from __future__ import annotations

import shutil
from pathlib import Path


def delete_input_folder(path: Path) -> tuple[bool, str]:
    """Move to recycle bin when send2trash is installed; otherwise delete normally."""
    try:
        from send2trash import send2trash  # type: ignore

        send2trash(str(path))
        return True, "Moved input folder to the recycle bin."
    except ImportError:
        shutil.rmtree(path)
        return True, "Permanently deleted input folder. This is not secure deletion."
    except Exception as exc:
        return False, str(exc)
