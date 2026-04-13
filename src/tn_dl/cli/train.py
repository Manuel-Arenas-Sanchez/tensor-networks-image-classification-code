from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tn_dl.config import load_experiment_config, load_runtime_config, resolve_experiment_config
from tn_dl.training import train


def main() -> None:
    args = _build_parser().parse_args()
    experiment = load_experiment_config(args.config)
    runtime = load_runtime_config(args.runtime) if args.runtime is not None else experiment.runtime
    resolved = resolve_experiment_config(
        experiment,
        runtime,
        device_override=args.device,
        output_dir_override=args.output_dir,
    )
    if args.resume is not None:
        resolved = replace(
            resolved,
            training=replace(resolved.training, resume_checkpoint=args.resume),
        )
    artifacts = train(resolved)
    print(f"Run directory: {artifacts.run_dir}")
    print(f"Best checkpoint: {artifacts.best_checkpoint}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tensor-network image classifier.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    parser.add_argument("--runtime", type=Path, help="Path to the runtime YAML.")
    parser.add_argument("--device", type=str, help="Device override: auto, cpu or cuda.")
    parser.add_argument("--output-dir", type=Path, help="Override the run root directory.")
    parser.add_argument("--resume", type=Path, help="Checkpoint path to resume from.")
    return parser


if __name__ == "__main__":
    main()
