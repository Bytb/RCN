#!/bin/bash -l
#SBATCH -J ppi-cp-full-vs-cyclebreak
#SBATCH -o ppi_cp_full_vs_cyclebreak_%A_%a.out
#SBATCH -e ppi_cp_full_vs_cyclebreak_%A_%a.err
#SBATCH -p standard
#SBATCH --time=130:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G

# Array tasks (4 total):
#   0-1 = full               (2 seeds)
#   2-3 = cyclebreak_shuffle (2 seeds)
#SBATCH --array=0-3%4

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

# seeds handled via --seed-index:
#   0 -> seed 120
#   1 -> seed 42
SEED_INDEX=$(( TASK_ID % 2 ))

VARIANT=""
if [ "${TASK_ID}" -le 1 ]; then
  VARIANT="full"
else
  VARIANT="cyclebreak_shuffle"
fi

echo "[MODE] Full vs Cyclebreak: --variant ${VARIANT} --seed-index ${SEED_INDEX} (0->120, 1->42)"

#Runs FullVsCyclebreak_ArrayRunner.py
srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK} \
  python -u -m RCN_Simulations.FullVsCyclebreak_ArrayRunner \
    --variant "${VARIANT}" \
    --seed-index "${SEED_INDEX}"

echo "Task ${TASK_ID} done at $(date)"

conda deactivate || true
exit 0
