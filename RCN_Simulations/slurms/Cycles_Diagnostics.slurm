#!/bin/bash -l
#SBATCH -J ppi-cycle-diagnostics
#SBATCH -o ppi_cycle_diagnostics_%A_%a.out
#SBATCH -e ppi_cycle_diagnostics_%A_%a.err
#SBATCH -p standard
#SBATCH --time=02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G

# Array tasks (2 seeds):
#   0 -> seed 120
#   1 -> seed 42
#SBATCH --array=0-1%2
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

# Cycle-break parameters (same defaults as your original script)
export SWAP_MULT=${SWAP_MULT:-10}
export RNBRW_WALKS=${RNBRW_WALKS:-6}
export RNBRW_WALKLEN=${RNBRW_WALKLEN:-8}
export RNBRW_STARTS=${RNBRW_STARTS:-4000}

cd "$REPO_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
echo "Running diagnostics array task index=${TASK_ID}"
echo "[PARAMS] SWAP_MULT=${SWAP_MULT} RNBRW_WALKS=${RNBRW_WALKS} RNBRW_WALKLEN=${RNBRW_WALKLEN} RNBRW_STARTS=${RNBRW_STARTS}"

SEED_INDEX="${TASK_ID}"
echo "[MODE] PPI Cycle Diagnostics: --seed-index ${SEED_INDEX} (0->120, 1->42)"

#Run Cycle_Diagnostics.py
srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK} \
  python -u -m RCN_Simulations.Cycle_Diagnostics \
    --seed-index "${SEED_INDEX}"

echo "Task ${TASK_ID} done at $(date)"

conda deactivate || true
exit 0
