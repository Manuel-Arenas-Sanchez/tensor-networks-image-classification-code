from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from tn_dl.config import load_runtime_config


def main() -> None:
    args = _build_parser().parse_args()
    runtime = load_runtime_config(args.runtime)
    partition = os.getenv("SLURM_PARTITION", runtime.slurm.partition)
    time_limit = os.getenv("SLURM_TIME", runtime.slurm.time)
    memory = os.getenv("SLURM_MEMORY", runtime.slurm.memory)
    gpus = int(os.getenv("SLURM_GPUS", str(runtime.slurm.gpus)))
    cpus = int(os.getenv("SLURM_CPUS_PER_TASK", str(runtime.slurm.cpus_per_task)))
    account = os.getenv("SLURM_ACCOUNT", runtime.slurm.account or "")
    project_root = args.project_root.resolve()

    wrapped_command = (
        f"cd {shlex.quote(str(project_root))} && "
        f"{shlex.quote(runtime.python_executable)} -m tn_dl.cli.train "
        f"--config {shlex.quote(str(args.config))} --runtime {shlex.quote(str(args.runtime))}"
    )
    command = [
        "sbatch",
        f"--job-name={runtime.slurm.job_name_prefix}-{args.config.stem}",
        f"--chdir={project_root}",
        f"--partition={partition}",
        f"--time={time_limit}",
        f"--mem={memory}",
        f"--cpus-per-task={cpus}",
        f"--gres=gpu:{gpus}",
    ]
    if account:
        command.append(f"--account={account}")
    command.append(f"--wrap={wrapped_command}")
    subprocess.run(command, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a training run to SLURM.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    parser.add_argument("--runtime", type=Path, required=True, help="Path to the runtime YAML.")
    parser.add_argument("--project-root", type=Path, required=True, help="Project root directory.")
    return parser


if __name__ == "__main__":
    main()
