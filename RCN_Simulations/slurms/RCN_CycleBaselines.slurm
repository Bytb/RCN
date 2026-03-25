#!/bin/bash -l
#SBATCH -J ppi-cp-cycle-baselines
#SBATCH -o ppi_cp_cycle_baselines_%A_%a.out
#SBATCH -e ppi_cp_cycle_baselines_%A_%a.err
#SBATCH -p standard
#SBATCH --time=130:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G

# Array tasks:
#   0-1   = full                  (2 seeds)
#   2-3   = uniform               (2 seeds)  (w=1)
#   4-5   = shuffled              (2 seeds)  (permute RNBRW weights across original edges)
#   6-7   = degree_only           (2 seeds)
#   8-9   = cyclebreak_shuffle    (2 seeds)  (RNBRW on rewired graph -> shuffle-assign to original edges)
#   10-11 = cyclebreak_degreepair (2 seeds)  (RNBRW on rewired graph -> assign by degree-pair bins)
#SBATCH --array=0-11%12

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
elif [ "${TASK_ID}" -le 3 ]; then
  VARIANT="uniform"
elif [ "${TASK_ID}" -le 5 ]; then
  VARIANT="shuffled"
elif [ "${TASK_ID}" -le 7 ]; then
  VARIANT="degree_only"
elif [ "${TASK_ID}" -le 9 ]; then
  VARIANT="cyclebreak_shuffle"
else
  VARIANT="cyclebreak_degreepair"
fi

echo "[MODE] Cycle baselines: --variant ${VARIANT} --seed-index ${SEED_INDEX} (0->120, 1->42)"

#Runs RCN_CycleBaselines.py
srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK} \
  python -u -m RCN_Simulations.RCN_CycleBaselines \
    --variant "${VARIANT}" \
    --seed-index "${SEED_INDEX}"

echo "Task ${TASK_ID} done at $(date)"

conda deactivate || true
exit 0
