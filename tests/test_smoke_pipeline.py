from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from tn_dl.config import (
    DataConfig,
    ExperimentConfig,
    IOConfig,
    ModelConfig,
    RuntimeConfig,
    SlurmConfig,
    TrainingConfig,
)
from tn_dl.data.registry import DATASET_REGISTRY
from tn_dl.training import evaluate, train


def _build_fake_dataset(_: DataConfig, train: bool) -> TensorDataset:
    generator = torch.Generator().manual_seed(123 if train else 456)
    images = torch.rand(12, 1, 28, 28, generator=generator)
    labels = torch.randint(0, 10, (12,), generator=generator)
    return TensorDataset(images, labels)


def test_train_and_evaluate_smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(DATASET_REGISTRY, "mnist", _build_fake_dataset)
    config = ExperimentConfig(
        name="smoke",
        seed=3,
        device="cpu",
        data=DataConfig(
            dataset="mnist",
            data_dir=tmp_path / "data",
            batch_size=4,
            num_workers=0,
            smoke_samples=8,
            download=False,
        ),
        model=ModelConfig(
            embedding_name="trig",
            layer_name="mps",
            input_shape=(1, 28, 28),
            local_dim=2,
            bond_dim=3,
            n_classes=10,
        ),
        training=TrainingConfig(
            epochs=1,
            learning_rate=0.01,
            weight_decay=0.0,
            log_every_n_steps=1,
        ),
        io=IOConfig(
            root_dir=tmp_path / "runs",
            checkpoint_name="best.pt",
            metrics_name="metrics.csv",
        ),
        runtime=RuntimeConfig(
            name="local",
            output_root=tmp_path / "runs",
            slurm=SlurmConfig(),
        ),
    )

    artifacts = train(config)
    metrics = evaluate(config, artifacts.best_checkpoint)

    assert artifacts.run_dir.exists()
    assert artifacts.best_checkpoint.exists()
    assert (artifacts.run_dir / "config.yaml").exists()
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.loss >= 0.0
