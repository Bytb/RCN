#!/bin/bash -l
#SBATCH -J ppi-cp-poc-plus-ksweep
#SBATCH -o ppi_cp_poc_plus_ksweep_%A_%a.out
#SBATCH -e ppi_cp_poc_plus_ksweep_%A_%a.err
#SBATCH -p standard
#SBATCH --time=130:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G

# Array tasks: 0-5 = PoC runs (3 configs x 2 seeds), 6-7 = K-sweep runs (2 seeds)
#SBATCH --array=0-7%8

#SBATCH --open-mode=truncate   # avoids appending across submissions

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

# 0-5: PoC overlap runs (existing runner expects --index 0..5)
if [ "${TASK_ID}" -le 5 ]; then
  echo "[MODE] PoC overlap run: --index ${TASK_ID}"
  srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK} \
    python -u -m RCN_Simulations.OverlapPoC_ArrayRunner --index "${TASK_ID}"

# 6-7: K-sweep runs (one per seed)
# We map: 6 -> seed index 0 (120), 7 -> seed index 1 (42)
#Runs KSweep_ArrayRunner.py
else
  KS_IDX=$(( TASK_ID - 6 ))
  if [ "${KS_IDX}" -lt 0 ] || [ "${KS_IDX}" -gt 1 ]; then
    echo "[ERROR] K-sweep index out of range: KS_IDX=${KS_IDX}" >&2
    exit 2
  fi
  echo "[MODE] K-sweep run: --seed-index ${KS_IDX}  (0->120, 1->42)"
  srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK} \
    python -u -m RCN_Simulations.KSweep_ArrayRunner --seed-index "${KS_IDX}"
fi

echo "Task ${TASK_ID} done at $(date)"

conda deactivate || true
exit 0
