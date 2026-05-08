#!/bin/bash
#SBATCH --job-name=ChemEq_2exp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ChemEq_2exp_%j.out
#SBATCH --error=logs/ChemEq_2exp_%j.err

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting at: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Si usas entorno virtual, descomenta y ajusta esto:
source .venv/bin/activate

python -m tn_dl.cli.train --config configs/experiments/ChemEq_exp1.yaml \
  > logs/ChemEq_exp1_${SLURM_JOB_ID}.out \
  2> logs/ChemEq_exp1_${SLURM_JOB_ID}.err &

PID1=$!

python -m tn_dl.cli.train --config configs/experiments/ChemEq_exp2.yaml \
  > logs/ChemEq_exp2_${SLURM_JOB_ID}.out \
  2> logs/ChemEq_exp2_${SLURM_JOB_ID}.err &

PID2=$!

echo "Launched exp1 with PID $PID1"
echo "Launched exp2 with PID $PID2"

wait $PID1
STATUS1=$?

wait $PID2
STATUS2=$?

echo "Experiment 1 finished with status $STATUS1 at $(date)"
echo "Experiment 2 finished with status $STATUS2 at $(date)"

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
    echo "At least one experiment failed."
    exit 1
fi

echo "Both experiments finished correctly at: $(date)"