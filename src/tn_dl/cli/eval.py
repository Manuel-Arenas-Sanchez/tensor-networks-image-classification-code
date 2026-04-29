from __future__ import annotations

import argparse
from pathlib import Path

from tn_dl.config import load_experiment_config, load_runtime_config, resolve_experiment_config
from tn_dl.training import evaluate


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
    metrics = evaluate(resolved, checkpoint_path=args.checkpoint)
    print(f"Loss: {metrics.loss:.6f}")
    print(f"Accuracy: {metrics.accuracy:.4f}")

    if args.diagnostics:
        from tn_dl.data import build_dataloaders
        from tn_dl.evaluation import write_classification_diagnostics
        from tn_dl.models import build_model
        from tn_dl.utils.device import resolve_device

        device = resolve_device(resolved.device)
        dataloaders = build_dataloaders(resolved.data)
        model = build_model(resolved.model).to(device)

        from tn_dl.training.engine import _prime_model, _load_checkpoint
        _prime_model(model, resolved, device)
        _load_checkpoint(
            checkpoint_path = args.checkpoint,
            model = model,
            optimizer = None,
            scheduler = None,
            device = device,
        )

        output_dir = args.checkpoint.parent / "diagnostics"
        eval_loader = dataloaders.eval_loader

        write_classification_diagnostics(
            model = model,
            dataloader = eval_loader,
            device = device,
            n_classes = resolved.model.n_classes,
            output_dir = output_dir,
        )
        print(f"Diagnostics saved to: {output_dir}")

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a tensor-network image classifier.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    parser.add_argument("--runtime", type=Path, help="Path to the runtime YAML.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--device", type=str, help="Device override: auto, cpu or cuda.")
    parser.add_argument("--output-dir", type=Path, help="Override the run root directory.")
    parser.add_argument("--diagnostics", action = "store_true", help = "Generate per-class diagnostics.")
    return parser


if __name__ == "__main__":
    main()
