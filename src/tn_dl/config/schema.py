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


@dataclass(slots=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    weight_decay: float = 0.0
    log_every_n_steps: int = 10
    resume_checkpoint: Path | None = None


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
