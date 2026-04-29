from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tn_dl.config import load_experiment_config
from tn_dl.evaluation import run_data_audit


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S:",
    )
    args = _build_parser().parse_args()
    config = load_experiment_config(args.config)

    output_dir = args.output_dir if args.output_dir is not None else Path("audit_output")

    run_data_audit(
        data_config=config.data,
        output_dir=output_dir,
        seed=config.seed,
    )
    print(f"Audit saved to: {output_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit data splits and augmentation.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML.")
    parser.add_argument("--output-dir", type=Path, help="Where to save audit results.")
    return parser


if __name__ == "__main__":
    main()