#!/bin/bash -l
#SBATCH -J combine-mechanisms
#SBATCH -o combine_mechanisms_%j.out
#SBATCH -e combine_mechanisms_%j.err
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --open-mode=truncate

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

echo "[MODE] Combining mechanism controls CSVs (raw + mean±std summary)"

#Runs combine_Cycle_Baselines.py
python -u -m RCN_Simulations.combine_Cycle_Baselines \
  --partial-subpath "logs/PPI_CP_OverlapPoC_Baselines/partial" \
  --out-dir-subpath "logs/PPI_CP_OverlapPoC_Baselines" \
  --pattern "mechanism_*_seed*.csv" \
  --raw-outfile "combined_mechanism_controls_raw.csv" \
  --summary-outfile "combined_mechanism_controls_summary.csv"

echo "Done at $(date)"

conda deactivate || true
exit 0
