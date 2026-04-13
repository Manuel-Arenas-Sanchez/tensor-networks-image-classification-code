from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from tn_dl.config.schema import (
    DataConfig,
    ExperimentConfig,
    IOConfig,
    ModelConfig,
    RuntimeConfig,
    SlurmConfig,
    TrainingConfig,
)


def load_experiment_config(path: Path) -> ExperimentConfig:
    payload = _read_yaml(path)
    return ExperimentConfig(
        name=_require_str(payload, "name"),
        seed=_require_int(payload, "seed"),
        device=_require_str(payload, "device"),
        data=_parse_data_config(_require_mapping(payload, "data")),
        model=_parse_model_config(_require_mapping(payload, "model")),
        training=_parse_training_config(_require_mapping(payload, "training")),
        io=_parse_io_config(_require_mapping(payload, "io")),
        runtime=_parse_runtime_config(payload.get("runtime", {})),
    )


def load_runtime_config(path: Path) -> RuntimeConfig:
    return _parse_runtime_config(_read_yaml(path))


def resolve_experiment_config(
    experiment: ExperimentConfig,
    runtime: RuntimeConfig | None,
    device_override: str | None = None,
    output_dir_override: Path | None = None,
) -> ExperimentConfig:
    resolved_runtime = runtime if runtime is not None else experiment.runtime
    resolved_root = output_dir_override or experiment.io.root_dir
    if output_dir_override is None and runtime is not None:
        resolved_root = runtime.output_root
    return replace(
        experiment,
        device=device_override or experiment.device,
        io=replace(experiment.io, root_dir=resolved_root),
        runtime=resolved_runtime,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def _parse_data_config(payload: dict[str, Any]) -> DataConfig:
    return DataConfig(
        dataset=_require_str(payload, "dataset"),
        data_dir=Path(_require_str(payload, "data_dir")),
        batch_size=_require_int(payload, "batch_size"),
        num_workers=int(payload.get("num_workers", 0)),
        smoke_samples=_optional_int(payload.get("smoke_samples")),
        download=bool(payload.get("download", True)),
    )


def _parse_model_config(payload: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        embedding_name=_require_str(payload, "embedding_name"),
        layer_name=_require_str(payload, "layer_name"),
        input_shape=_require_int_tuple(payload, "input_shape", expected_length=3),
        local_dim=_require_int(payload, "local_dim"),
        bond_dim=_require_int(payload, "bond_dim"),
        n_classes=_require_int(payload, "n_classes"),
        out_position=_optional_int(payload.get("out_position")),
        boundary=str(payload.get("boundary", "obc")),
    )


def _parse_training_config(payload: dict[str, Any]) -> TrainingConfig:
    resume_checkpoint = payload.get("resume_checkpoint")
    return TrainingConfig(
        epochs=_require_int(payload, "epochs"),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload.get("weight_decay", 0.0)),
        log_every_n_steps=int(payload.get("log_every_n_steps", 10)),
        resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint else None,
    )


def _parse_io_config(payload: dict[str, Any]) -> IOConfig:
    return IOConfig(
        root_dir=Path(_require_str(payload, "root_dir")),
        checkpoint_name=str(payload.get("checkpoint_name", "best.pt")),
        metrics_name=str(payload.get("metrics_name", "metrics.csv")),
        resolved_config_name=str(payload.get("resolved_config_name", "config.yaml")),
    )


def _parse_runtime_config(payload: dict[str, Any]) -> RuntimeConfig:
    slurm_payload = _require_mapping(payload, "slurm", allow_missing=True)
    return RuntimeConfig(
        name=str(payload.get("name", "local")),
        output_root=Path(str(payload.get("output_root", "runs"))),
        python_executable=str(payload.get("python_executable", "python")),
        slurm=SlurmConfig(
            partition=str(slurm_payload.get("partition", "gpu")),
            time=str(slurm_payload.get("time", "00:30:00")),
            memory=str(slurm_payload.get("memory", "16G")),
            gpus=int(slurm_payload.get("gpus", 1)),
            cpus_per_task=int(slurm_payload.get("cpus_per_task", 4)),
            account=_optional_str(slurm_payload.get("account")),
            job_name_prefix=str(slurm_payload.get("job_name_prefix", "tn-dl")),
        ),
    )


def _require_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        return {} if allow_missing else _raise_missing(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected '{key}' to be a string.")
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected '{key}' to be an integer.")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("Expected an integer or null value.")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected a string or null value.")
    return value


def _require_int_tuple(
    payload: dict[str, Any],
    key: str,
    *,
    expected_length: int,
) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"Expected '{key}' to be a list with {expected_length} integers.")
    if not all(isinstance(item, int) for item in value):
        raise ValueError(f"Expected '{key}' to contain only integers.")
    return tuple(value)


def _raise_missing(key: str) -> dict[str, Any]:
    raise ValueError(f"Missing '{key}' in configuration.")
