import csv
import datetime
from pathlib import Path
from typing import Optional, Any


class ExperimentLogger:
    """
    Create logging CSV file up‑front and append rows to it as each experiment finishes.
    """

    def __init__(self, logs_dir: Path, job_id: Optional[str] = None) -> None:
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        job_suffix = f"_{job_id}" if job_id else ""
        filename = f"experiment_results_{timestamp}{job_suffix}.csv"
        self.log_file = self.logs_dir / filename

        self._fieldnames: Optional[list[str]] = None

    def _init_header(self, fieldnames: list[str]) -> None:
        with self.log_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        self._fieldnames = fieldnames

    def _coerce(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return a copy with hypothesis_label formatted to int."""
        if "hypothesis_label" in record:
            record = {**record, "hypothesis_label": int(record["hypothesis_label"])}
        return record

    def append(self, record: dict[str, Any]) -> None:
        if self._fieldnames is None:
            self._init_header(list(record.keys()))

        with self.log_file.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames).writerow(
                self._coerce(record)
            )

    def append_many(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        if self._fieldnames is None:
            self._init_header(list(records[0].keys()))

        with self.log_file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerows(map(self._coerce, records))
