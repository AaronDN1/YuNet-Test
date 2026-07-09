from __future__ import annotations

from datetime import datetime
from pathlib import Path


class ProcessingLog:
    """Accumulates run stats and writes a single end-of-run report file."""

    def __init__(self, output_dir: Path, input_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = output_dir / f"face_anonymization_report_{timestamp}.txt"
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._started_at = datetime.now()
        self._failed: list[tuple[str, str]] = []
        self._anonymized = 0
        self._quarantined = 0
        self._faces_detected = 0
        self._total_files = 0
        self._mode = ""
        self._detector_note = ""

    @property
    def failed_count(self) -> int:
        return len(self._failed)

    @property
    def processed_count(self) -> int:
        return self._anonymized + self._quarantined

    def configure(self, total_files: int, mode: str, detector_note: str) -> None:
        self._total_files = total_files
        self._mode = mode
        self._detector_note = detector_note

    def record_success(self, source: Path, destination: Path, faces: int, quarantined: bool = False) -> None:
        _ = (source, destination)
        if quarantined:
            self._quarantined += 1
        else:
            self._anonymized += 1
            self._faces_detected += faces

    def record_failure(self, source: Path, error: str) -> None:
        # Keep a short error for the report; drop long tracebacks from the summary.
        short_error = error.splitlines()[0] if error else "Unknown error"
        self._failed.append((str(source), short_error))

    def finish(self, deletion_performed: bool) -> str:
        ended_at = datetime.now()
        elapsed = ended_at - self._started_at
        total_seconds = max(0.0, elapsed.total_seconds())
        processed = self.processed_count
        images_per_second = processed / total_seconds if total_seconds > 0 else 0.0

        lines = [
            "Local Face Anonymization Report",
            "=" * 36,
            f"Start time:              {self._started_at.isoformat(timespec='seconds')}",
            f"End time:                {ended_at.isoformat(timespec='seconds')}",
            f"Runtime:                 {_format_duration(total_seconds)}",
            f"Throughput:              {images_per_second:.2f} images/sec",
            "",
            f"Input folder:            {self._input_dir}",
            f"Output folder:           {self._output_dir}",
            f"Anonymization mode:      {self._mode or 'n/a'}",
            f"Detector:                {self._detector_note or 'n/a'}",
            "",
            "Results",
            "-" * 36,
            f"Images discovered:       {self._total_files}",
            f"Images anonymized:       {self._anonymized}",
            f"Images quarantined:      {self._quarantined}",
            f"Images failed:           {len(self._failed)}",
            f"Faces anonymized:        {self._faces_detected}",
            f"Input deletion performed: {'yes' if deletion_performed else 'no'}",
        ]
        if self._failed:
            lines.extend(["", "Failed files", "-" * 36])
            for path, error in self._failed:
                lines.append(f"- {path}")
                lines.append(f"  {error}")

        report_text = "\n".join(lines) + "\n"
        self.path.write_text(report_text, encoding="utf-8")
        return report_text


def _format_duration(total_seconds: float) -> str:
    seconds = int(round(total_seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s ({total_seconds:.1f}s)"
    if minutes:
        return f"{minutes}m {secs}s ({total_seconds:.1f}s)"
    return f"{total_seconds:.1f}s"
