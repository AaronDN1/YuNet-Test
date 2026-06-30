from __future__ import annotations

from datetime import datetime
from pathlib import Path


class ProcessingLog:
    def __init__(self, output_dir: Path, input_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = output_dir / f"face_anonymization_log_{timestamp}.txt"
        self._failed: list[tuple[str, str]] = []
        self._processed = 0
        self._quarantined = 0
        self._write(
            [
                "Local Face Anonymization Log",
                f"Start time: {datetime.now().isoformat(timespec='seconds')}",
                f"Input folder: {input_dir}",
                f"Output folder: {output_dir}",
                "",
            ]
        )

    @property
    def failed_count(self) -> int:
        return len(self._failed)

    @property
    def processed_count(self) -> int:
        return self._processed

    def record_success(self, source: Path, destination: Path, faces: int, quarantined: bool = False) -> None:
        self._processed += 1
        if quarantined:
            self._quarantined += 1
        self._write(
            [
                f"OK: {source}",
                f"  Output: {destination}",
                f"  Faces detected: {faces}",
                f"  Quarantined for manual review: {quarantined}",
            ]
        )

    def record_failure(self, source: Path, error: str) -> None:
        self._failed.append((str(source), error))
        self._write([f"FAILED: {source}", f"  Error: {error}"])

    def finish(self, deletion_performed: bool) -> None:
        lines = [
            "",
            f"End time: {datetime.now().isoformat(timespec='seconds')}",
            f"Files processed: {self._processed}",
            f"Files quarantined (no face detected): {self._quarantined}",
            f"Failed files: {len(self._failed)}",
            f"Input deletion performed: {deletion_performed}",
        ]
        if self._failed:
            lines.append("")
            lines.append("Failed file list:")
            for path, error in self._failed:
                lines.append(f"- {path}: {error}")
        self._write(lines)

    def _write(self, lines: list[str]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"{line}\n")
