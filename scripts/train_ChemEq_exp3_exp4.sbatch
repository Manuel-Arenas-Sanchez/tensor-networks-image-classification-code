#!/bin/bash
#SBATCH --job-name=ChemEq_3_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ChemEq_3_4_%j.out
#SBATCH --error=logs/ChemEq_3_4_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting at: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

nvidia-smi || true

source .venv/bin/activate

python -u -m tn_dl.cli.train --config configs/experiments/ChemEq_exp3.yaml \
  > logs/ChemEq_exp3_${SLURM_JOB_ID}.out \
  2> logs/ChemEq_exp3_${SLURM_JOB_ID}.err &

PID1=$!

sleep 10

python -u -m tn_dl.cli.train --config configs/experiments/ChemEq_exp4.yaml \
  > logs/ChemEq_exp4_${SLURM_JOB_ID}.out \
  2> logs/ChemEq_exp4_${SLURM_JOB_ID}.err &

PID2=$!

echo "Launched exp3 with PID $PID1"
echo "Launched exp4 with PID $PID2"

wait $PID1
STATUS1=$?

wait $PID2
STATUS2=$?

echo "Experiment 3 finished with status $STATUS1 at $(date)"
echo "Experiment 4 finished with status $STATUS2 at $(date)"

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
    echo "At least one experiment failed."
    exit 1
fi

echo "Both experiments finished correctly at: $(date)"
