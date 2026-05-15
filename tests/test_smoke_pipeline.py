from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tn_dl.config import (
    DataConfig,
    ExperimentConfig,
    IOConfig,
    ModelConfig,
    RuntimeConfig,
    SlurmConfig,
    TrainingConfig,
)
from tn_dl.data import DataLoaders
from tn_dl.training import engine, evaluate, train


def _build_fake_dataset(seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(12, 1, 28, 28, generator=generator)
    labels = torch.randint(0, 10, (12,), generator=generator)
    return TensorDataset(images, labels)


def _build_fake_dataloaders(data_config: DataConfig) -> DataLoaders:
    return DataLoaders(
        train_loader=DataLoader(
            _build_fake_dataset(seed=123),
            batch_size=data_config.batch_size,
        ),
        eval_loader=DataLoader(
            _build_fake_dataset(seed=456),
            batch_size=data_config.batch_size,
        ),
        test_loader=None,
    )


def test_train_and_evaluate_smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "build_dataloaders", _build_fake_dataloaders)
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
    assert (artifacts.run_dir / "metric_curves.png").exists()
    assert (artifacts.run_dir / "confidence_curves.png").exists()
    assert (artifacts.run_dir / "diagnostics" / "summary_metrics.csv").exists()
    assert (artifacts.run_dir / "diagnostics" / "precision_recall_curve.png").exists()
    assert (artifacts.run_dir / "diagnostics" / "f1_confidence_curve.png").exists()
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.loss >= 0.0
