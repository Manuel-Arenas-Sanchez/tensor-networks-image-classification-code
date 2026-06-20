from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def create_run_dir(root_dir: Path, experiment_name: str) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root_dir / f"{timestamp}-{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_yaml_file(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_serializable(payload), handle, sort_keys=False)


def write_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value
