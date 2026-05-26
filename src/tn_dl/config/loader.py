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
    image_size = int(payload.get("image_size", 224))
    crop_margin = float(payload.get("crop_margin", 0.05))

    if image_size <= 0:
        raise ValueError("Expected 'image_size' to be > 0.")
    if crop_margin < 0.0:
        raise ValueError("Expected 'crop_margin' to be >= 0.")
    
    return DataConfig(
        dataset=_require_str(payload, "dataset"),
        data_dir=Path(_require_str(payload, "data_dir")),
        batch_size=_require_int(payload, "batch_size"),
        num_workers=int(payload.get("num_workers", 0)),
        smoke_samples=_optional_int(payload.get("smoke_samples")),
        download=bool(payload.get("download", True)),
        image_size = image_size,
        crop_margin = crop_margin,
        preserve_aspect_ratio = bool(payload.get("preserve_aspect_ratio", False)),
        augmentation_name=str(payload.get("augmentation_name", "basic")),
        rotation_degrees=float(payload.get("rotation_degrees", 0.0)),
        random_erasing_p=float(payload.get("random_erasing_p", 0.0)),
        normalization_name=_parse_normalization_name(payload.get("normalization_name")),
    )


def _parse_model_config(payload: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        embedding_name=_require_str(payload, "embedding_name"),
        layer_name=_require_str(payload, "layer_name"),
        input_shape=_require_input_shape(payload, "input_shape"),
        local_dim=_require_int(payload, "local_dim"),
        bond_dim=_require_int(payload, "bond_dim"),
        n_classes=_require_int(payload, "n_classes"),
        out_position=_optional_int(payload.get("out_position")),
        boundary=str(payload.get("boundary", "obc")),
        patch_size=_optional_int(payload.get("patch_size")),
        linearization_order = _parse_linearization_order(payload.get("linearization_order")),
        aim_channels = _optional_int_with_default(
            payload.get("aim_channels"), 32, "aim_channels"
        ),
        num_aim_blocks = _optional_int_with_default(
            payload.get("num_aim_blocks"), 2, "num_aim_blocks"
        ),
        aim_expansion_ratio = _optional_int_with_default(
            payload.get("aim_expansion_ratio"), 3, "aim_expansion_ratio"
        ),
        aim_kernel_size = _optional_int_with_default(
            payload.get("aim_kernel_size"), 3, "aim_kernel_size"
        ),
        aim_use_ln = _optional_bool_with_default(payload.get("aim_use_ln"), True, "aim_use_ln"),
        head_type = _parse_head_type(payload.get("head_type")),
        head_hidden_dim = _optional_int_with_default(
            payload.get("head_hidden_dim"), 32, "head_hidden_dim"
        ),
        head_dropout = _parse_dropout(payload.get("head_dropout")),
    )

def _parse_training_config(payload: dict[str, Any]) -> TrainingConfig:
    resume_checkpoint = payload.get("resume_checkpoint")
    epochs = _require_int(payload, "epochs")
    warmup_epochs = _parse_nonnegative_int(
        payload.get("warmup_epochs"), 0, "warmup_epochs"
    )

    if warmup_epochs > epochs:
        raise ValueError("Expected 'warmup_epochs' to be <= 'epochs'.")

    return TrainingConfig(
        epochs=epochs,
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload.get("weight_decay", 0.0)),
        log_every_n_steps=int(payload.get("log_every_n_steps", 10)),
        resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint else None,
        warmup_name=_parse_warmup_name(payload.get("warmup_name")),
        warmup_epochs=warmup_epochs,
        warmup_start_factor=_parse_warmup_start_factor(
            payload.get("warmup_start_factor")
        ),
        regularization_name=_parse_regularization_name(
            payload.get("regularization_name")
        ),
        regularization_lambda=_parse_nonnegative_float(
            payload.get("regularization_lambda"), 0.0, "regularization_lambda"
        ),
        label_smoothing=_parse_label_smoothing(payload.get("label_smoothing")),
        scheduler_name=_optional_str(payload.get("scheduler_name")),
        scheduler_factor=float(payload.get("scheduler_factor", 0.5)),
        scheduler_patience=int(payload.get("scheduler_patience", 5)),
        scheduler_min_lr=float(payload.get("scheduler_min_lr", 1e-6)),
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

def _require_input_shape(payload: dict[str, Any], key: str) -> tuple[int, int, int]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Expected '{key}' to be a list with 3 integers.")
    if not all(isinstance(item, int) for item in value):
        raise ValueError(f"Expected '{key}' to contain only integers.")
    return value[0], value[1], value[2]

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

def _parse_linearization_order(value: Any) -> str:
    if value is None:
        return "zigzag"
    if not isinstance(value, str):
        raise ValueError("Expected 'linearization_order' to be a string.")
    normalized = value.lower()
    if normalized not in ("row", "zigzag"):
        raise ValueError("Expected 'linearization_order' to be 'row' or 'zigzag'.")
    return normalized

def _parse_normalization_name(value: Any) -> str:
    if value is None:
        return "imagenet"
    if not isinstance(value, str):
        raise ValueError("Expected 'normalization_name' to be a string.")
    normalized = value.lower()
    if normalized not in ("imagenet", "none"):
        raise ValueError("Expected 'normalization_name' to be 'imagenet' or 'none'.")
    return normalized

def _parse_head_type(value: Any) -> str:
    if value is None:
        return "linear"
    if not isinstance(value, str):
        raise ValueError("Expected 'head_type' to be a string.")
    normalized = value.lower()
    if normalized not in ("linear", "mlp"):
        raise ValueError("Expected 'head_type' to be 'linear' or 'mlp'.")
    return normalized

def _parse_dropout(value: Any) -> float:
    if value is None:
        return 0.0
    dropout = float(value)
    if not 0.0 <= dropout < 1.0:
        raise ValueError("Expected 'head_dropout' to be in the interval [0, 1).")
    return dropout

def _raise_missing(key: str) -> dict[str, Any]:
    raise ValueError(f"Missing '{key}' in configuration.")

def _optional_int_with_default(value: Any, default: int, key: str) -> int:
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValueError(f"Expected '{key}' to be an integer.")
    return value

def _optional_bool_with_default(value: Any, default: bool, key: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"Expected '{key}' to be a boolean.")
    return value

def _parse_nonnegative_int(value: Any, default: int, key: str) -> int:
    parsed = _optional_int_with_default(value, default, key)
    if parsed < 0:
        raise ValueError(f"Expected '{key}' to be >= 0.")
    return parsed

def _parse_warmup_start_factor(value: Any) -> float:
    if value is None:
        return 0.2
    factor = float(value)
    if not 0.0 < factor <= 1.0:
        raise ValueError("Expected 'warmup_start_factor' to be in the interval (0, 1].")
    return factor

def _parse_warmup_name(value: Any) -> str:
    if value is None:
        return "linear"
    if not isinstance(value, str):
        raise ValueError("Expected 'warmup_name' to be a string.")
    normalized = value.lower()
    if normalized not in ("none", "linear", "cosine"):
        raise ValueError("Expected 'warmup_name' to be 'none', 'linear', or 'cosine'.")
    return normalized

def _parse_regularization_name(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected 'regularization_name' to be a string.")
    normalized = value.lower()
    if normalized in ("none", "null"):
        return None
    if normalized not in ("l2",):
        raise ValueError("Expected 'regularization_name' to be None or 'l2'.")
    return normalized


def _parse_nonnegative_float(value: Any, default: float, key: str) -> float:
    if value is None:
        return default
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError(f"Expected '{key}' to be >= 0.")
    return parsed

def _parse_label_smoothing(value: Any) -> float:
    if value is None:
        return 0.0
    smoothing = float(value)
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("Expected 'label_smoothing' to be in the interval [0, 1).")
    return smoothing
