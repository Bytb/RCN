#!/bin/bash -l
#SBATCH -J combine-ppi-lambda-sens
#SBATCH -o combine_ppi_lambda_sens_%j.out
#SBATCH -e combine_ppi_lambda_sens_%j.err
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --open-mode=truncate

set -euo pipefail

REPO_DIR="path-to-repo-dir/RCN_Test"

echo "Host: $(hostname) | JobID: ${SLURM_JOB_ID:-NA} | Date: $(date)"
echo "[MODE] Combining PPI Lambda Sensitivity CSVs (raw + mean±std)"

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

#Run combine_LambdaSensitivit.py
python -u -m RCN_Simulations.combine_LambdaSensitivity \
  --partial-subpath "logs/PPI_LambdaSensitivity/partial" \
  --out-dir-subpath "logs/PPI_LambdaSensitivity" \
  --pattern "run_*_seed*.csv" \
  --raw-outfile "combined_lambda_sensitivity_raw.csv" \
  --summary-outfile "combined_lambda_sensitivity_summary.csv"

echo "Done at $(date)"

conda deactivate || true
exit 0
