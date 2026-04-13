from __future__ import annotations

from pathlib import Path

from tn_dl.config import (
    ExperimentConfig,
    RuntimeConfig,
    load_experiment_config,
    load_runtime_config,
    resolve_experiment_config,
)


def test_load_experiment_config_parses_nested_types(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: demo",
                "seed: 11",
                "device: auto",
                "data:",
                "  dataset: mnist",
                "  data_dir: data",
                "  batch_size: 8",
                "  num_workers: 0",
                "  smoke_samples: 16",
                "  download: true",
                "model:",
                "  embedding_name: trig",
                "  layer_name: mps",
                "  input_shape: [1, 28, 28]",
                "  local_dim: 2",
                "  bond_dim: 4",
                "  n_classes: 10",
                "training:",
                "  epochs: 2",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  log_every_n_steps: 1",
                "io:",
                "  root_dir: runs",
                "  checkpoint_name: best.pt",
                "  metrics_name: metrics.csv",
            ]
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert isinstance(config, ExperimentConfig)
    assert config.name == "demo"
    assert config.data.data_dir == Path("data")
    assert config.model.input_shape == (1, 28, 28)
    assert config.training.epochs == 2
    assert config.io.root_dir == Path("runs")


def test_load_runtime_config_and_resolve_overrides(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    runtime_path = tmp_path / "runtime.yaml"
    experiment_path.write_text(
        "\n".join(
            [
                "name: base",
                "seed: 7",
                "device: auto",
                "data:",
                "  dataset: fashion_mnist",
                "  data_dir: data",
                "  batch_size: 4",
                "  num_workers: 0",
                "  smoke_samples: 8",
                "  download: false",
                "model:",
                "  embedding_name: trig",
                "  layer_name: mps",
                "  input_shape: [1, 28, 28]",
                "  local_dim: 2",
                "  bond_dim: 3",
                "  n_classes: 10",
                "training:",
                "  epochs: 1",
                "  learning_rate: 0.01",
                "  weight_decay: 0.0",
                "  log_every_n_steps: 1",
                "io:",
                "  root_dir: runs",
                "  checkpoint_name: best.pt",
                "  metrics_name: metrics.csv",
            ]
        ),
        encoding="utf-8",
    )
    runtime_path.write_text(
        "\n".join(
            [
                "name: slurm",
                "output_root: cluster_runs",
                "slurm:",
                "  partition: gpu",
                "  time: '00:30:00'",
                "  memory: 8G",
                "  gpus: 1",
                "  cpus_per_task: 4",
                "  account: research",
            ]
        ),
        encoding="utf-8",
    )

    experiment = load_experiment_config(experiment_path)
    runtime = load_runtime_config(runtime_path)
    resolved = resolve_experiment_config(
        experiment,
        runtime,
        device_override="cpu",
        output_dir_override=tmp_path / "artifacts",
    )

    assert isinstance(runtime, RuntimeConfig)
    assert runtime.slurm.partition == "gpu"
    assert resolved.device == "cpu"
    assert resolved.io.root_dir == tmp_path / "artifacts"
    assert resolved.runtime.output_root == Path("cluster_runs")
