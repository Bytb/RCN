#!/bin/bash -l
#SBATCH -J combine
#SBATCH -o combine_%j.out
#SBATCH -e combine_%j.err
#SBATCH -p standard             
#SBATCH -c 1
#SBATCH --mem=384G
#SBATCH --time=130:00:00

set -euo pipefail

# --- Repo root ---
REPO_DIR="path-to-repo-dir/RCN_Test"

echo "Host: $(hostname) | JobID: ${SLURM_JOB_ID:-NA} | Date: $(date)"
echo "Partition: ${SLURM_JOB_PARTITION:-NA} | GRES: ${SLURM_JOB_GRES:-NA}"

module purge
module load miniforge/24.3.0-py3.11

# Temporarily relax nounset for conda activation (MKL env quirk)
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/$USER/.conda/envs/torch39
set -u

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

# Ensure Python can import the repo packages
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Optional CUDA sanity check
python - <<'PYCHK'
import torch, os, sys
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA build:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Repo on sys.path?", any(p.endswith("GNN-Advanced-Community-Detection") for p in sys.path))
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))
PYCHK

# Run from repo root and use -m to resolve package imports
# Runs combine_Hyper_Results.py
cd "$REPO_DIR"
echo "Running: python -u -m RCN_Simulations.combine_Hyper_Results"
python -u -m RCN_Simulations.combine_Hyper_Results

echo "Done at $(date)"
