"""Epoch-level metric logger."""

import os
import csv
import json
from pathlib import Path
from typing import Dict, Any, List


class TrainingLogger:
    """Logs epoch metrics to CSV and JSONL."""

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.fieldnames: List[str] = []
        self._csv_file = None
        self._writer = None

    def log(self, metrics: Dict[str, Any]):
        """Write an epoch's metrics to CSV and JSONL."""
        # JSONL
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # CSV
        keys = list(metrics.keys())
        if self._writer is None:
            self.fieldnames = keys
            self._csv_file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames)
            self._writer.writeheader()

        self._writer.writerow(metrics)
        self._csv_file.flush()

    def close(self):
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()

