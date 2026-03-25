#!/bin/bash -l
#SBATCH -J ppi-array
#SBATCH -o ppi_array_%A_%a.out
#SBATCH -e ppi_array_%A_%a.err
#SBATCH -p standard
#SBATCH --time=130:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G
#SBATCH --exclusive
#SBATCH --array=0-107%32
#SBATCH --open-mode=truncate   # avoids appending across submissions

set -euo pipefail

REPO_DIR="path-to-repo-dir/RCN_Test"
mkdir -p "logs/slurm/${SLURM_JOB_ID}"

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

# ensure configs exist
if [ ! -f "$REPO_DIR/RCN_Simulations/configs/PPI_sweep2_configs.csv" ]; then
  echo "[INFO] Generating configs..."
  python -u "$REPO_DIR/RCN_Simulations/configs/make_ppi_configs.py"
fi

cd "$REPO_DIR"

echo "Running task index=${SLURM_ARRAY_TASK_ID}"
srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK} \
  python -u -m RCN_Simulations.HyperParam_ArrayRunner --index "${SLURM_ARRAY_TASK_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID} done at $(date)"

conda deactivate || true
exit 0
