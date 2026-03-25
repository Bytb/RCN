#!/bin/bash -l
#SBATCH -J ppi-cp-full-vs-cyclebreak-combine
#SBATCH -o ppi_cp_full_vs_cyclebreak_combine_%j.out
#SBATCH -e ppi_cp_full_vs_cyclebreak_combine_%j.err
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_DIR="path-to-repo-dir/RCN_Test"

echo "Host: $(hostname) | JobID: ${SLURM_JOB_ID:-NA} | Date: $(date)"

module purge
module load miniforge/24.3.0-py3.11

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/$USER/.conda/envs/torch39
set -u

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "[MODE] Combining Full vs Cyclebreak CSVs + computing deltas"

#Runs the combine_full_vs_cyclebreak.py file
python -u -m RCN_Simulations.combine_full_vs_cyclebreak

echo "Done at $(date)"

conda deactivate || true
exit 0
