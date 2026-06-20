from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    dataset: str
    data_dir: Path
    batch_size: int
    num_workers: int = 0
    smoke_samples: int | None = None
    download: bool = True
    image_size: int = 224
    crop_margin: float = 0.05
    preserve_aspect_ratio: bool = False


@dataclass(slots=True)
class ModelConfig:
    embedding_name: str
    layer_name: str
    input_shape: tuple[int, int, int]
    local_dim: int
    bond_dim: int
    n_classes: int
    out_position: int | None = None
    boundary: str = "obc"
    patch_size: int | None = None
    linearization_order: str = "zigzag" # "row" o "zigzag"

    #Hiperparámetros específicos del front-end AIM
    aim_channels: int = 32
    num_aim_blocks: int = 2
    aim_expansion_ratio: int = 3
    aim_kernel_size: int = 3
    aim_use_ln: bool = True

    # Hiperparametros de la cabeza de clasificacion
    head_type: str = "linear" # "linear" o "mlp"
    head_hidden_dim: int = 32
    head_dropout: float = 0.0

    # Hiperparametros especificos de baselines de vision (ResNet50)
    pretrained: bool = False       # pesos ImageNet vs aleatorios (from scratch)
    freeze_backbone: bool = False  # entrenar solo la cabeza fc

@dataclass(slots=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    weight_decay: float = 0.0
    log_every_n_steps: int = 10
    resume_checkpoint: Path | None = None

    warmup_name: str = "linear"
    warmup_epochs: int = 0
    warmup_start_factor: float = 0.2

    regularization_name: str | None = None
    regularization_lambda: float = 0.0
    label_smoothing: float = 0.0

    scheduler_name: str | None = None
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6

    # Early stopping (opcional, desactivado por defecto con patience = 0)
    early_stopping_patience: int = 0
    early_stopping_metric: str = "eval_loss"  # "eval_loss" o "eval_accuracy"
    early_stopping_min_delta: float = 0.0


@dataclass(slots=True)
class IOConfig:
    root_dir: Path
    checkpoint_name: str = "best.pt"
    metrics_name: str = "metrics.csv"
    resolved_config_name: str = "config.yaml"


@dataclass(slots=True)
class SlurmConfig:
    partition: str = "gpu"
    time: str = "00:30:00"
    memory: str = "16G"
    gpus: int = 1
    cpus_per_task: int = 4
    account: str | None = None
    job_name_prefix: str = "tn-dl"


@dataclass(slots=True)
class RuntimeConfig:
    name: str = "local"
    output_root: Path = Path("runs")
    python_executable: str = "python"
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    io: IOConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
