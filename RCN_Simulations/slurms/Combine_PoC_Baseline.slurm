#!/bin/bash -l
#SBATCH -J baseline-ksweep-combine
#SBATCH -o baseline_ksweep_combine_%j.out
#SBATCH -e baseline_ksweep_combine_%j.err
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_DIR="path-to-repo-root/RCN_Test"

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

echo "[MODE] Combining baseline K-sweep CSVs + generating plots"

python -u -m RCN_Simulations.combine_baseline_ksweep_csvs_and_plot \
  --partial-subpath "logs/PPI_CP_OverlapPoC_Baselines/partial" \
  --out-dir-subpath "logs/PPI_CP_OverlapPoC_Baselines" \
  --outfile "combined_k_sweep_baselines.csv"

echo "Done at $(date)"

conda deactivate || true
exit 0
