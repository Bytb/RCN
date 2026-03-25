#!/bin/bash -l
#SBATCH -J ppi-lambda-sens
#SBATCH -o ppi_lambda_sens_%A_%a.out
#SBATCH -e ppi_lambda_sens_%A_%a.err
#SBATCH -p standard
#SBATCH --time=130:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G

# Array tasks (32 total):
#   tid = 0..31  ->  4 lambdas × 4 multipliers × 2 seeds
# Mapping (must match LambdaSensitivity_ArrayRunner.py):
#   lam_i  = tid // 8            -> 0..3  (lambda_mod, lambda_lap, lambda_contrast, lambda_orth)
#   rem    = tid % 8
#   mult_i = rem // 2            -> 0..3  (0.0, 0.5, 1.0, 2.0)
#   seed_i = rem % 2             -> 0..1  (seed 120, 42)
#SBATCH --array=0-31%32

#SBATCH --open-mode=truncate

set -euo pipefail

REPO_DIR="path-to-repo-dir/RCN_Test"
mkdir -p "$REPO_DIR/logs/slurm/${SLURM_JOB_ID}"

echo "Host: $(hostname) | JobID: ${SLURM_JOB_ID:-NA} | Task: ${SLURM_ARRAY_TASK_ID:-NA} | Date: $(date)"
echo "Partition: ${SLURM_JOB_PARTITION:-NA} | CPUs-per-task: ${SLURM_CPUS_PER_TASK:-NA}"

module purge
module load miniforge/24.3.0-py3.11

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/$USER/.conda/envs/torch39
set -u

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$REPO_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
echo "Running array task index=${TASK_ID}"

# Helpful echo of decoded mapping (same as runner):
LAM_KEYS=("lambda_mod" "lambda_lap" "lambda_contrast" "lambda_orth")
MULTS=("0.0" "0.5" "1.0" "2.0")
SEEDS=("120" "42")

LAM_I=$(( TASK_ID / 8 ))
REM=$(( TASK_ID % 8 ))
MULT_I=$(( REM / 2 ))
SEED_I=$(( REM % 2 ))

echo "[DECODE] lambda=${LAM_KEYS[$LAM_I]}  multiplier=${MULTS[$MULT_I]}  seed=${SEEDS[$SEED_I]}"

#Run LambdaSensitivity_ArrayRunner.py
srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK} \
  python -u RCN_Simulations/LambdaSensitivity_ArrayRunner.py

echo "Task ${TASK_ID} done at $(date)"

conda deactivate || true
exit 0
