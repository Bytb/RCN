#!/bin/bash
#SBATCH --job-name=calc_weights
#SBATCH --output=output_path_%j.txt
#SBATCH --error=error_path_%j.txt
#SBATCH --time=130:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=standard 
#SBATCH --mail-user=user
#SBATCH --array=0-99

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# --- Start of Revised Section ---
#This script will run the run_runbrw_batch.py file

set -euxo pipefail
echo "Host: $(hostname)  Date: $(date)"
echo "User: $(whoami)  Shell: $SHELL  Bash: $BASH_VERSION"

echo "=== Modules BEFORE purge ==="
module list || true

# Kill anything your dotfiles might have loaded
module --force purge

echo "=== Modules AFTER purge ==="
module list || true

# Prevent bash from auto-sourcing user init within the job
unset BASH_ENV

# Load Python via miniforge (recommended for Rivanna)
# Load Python 3.9 from available modules
module purge
# Load Python via miniforge (recommended for Rivanna)
echo "Loading Python via miniforge..."
module load miniforge/24.3.0-py3.11 || {
    echo "ERROR: Failed to load miniforge Python module."
    echo "Check available modules with 'module avail miniforge' or 'module spider miniforge'"
    exit 1
}

# Find Python executable (should find the one from miniforge now)
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "ERROR: No Python executable found even after loading miniforge."
    exit 1
fi

echo "Python executable: $(which $PYTHON_CMD)"
echo "Python version: $($PYTHON_CMD --version)" # Verify it's Python 3.11.x

# Set up a virtual environment
VENV_DIR="$HOME/venvs/rnbrw_env_py311" # Using a distinct name for the new venv
echo "Setting up virtual environment at $VENV_DIR"

# Create virtual environment if it doesn't exist (using the loaded miniforge python)
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating new virtual environment..."
    # Use the found Python command to create the venv
    $PYTHON_CMD -m venv "$VENV_DIR" || {
        echo "ERROR: Failed to create virtual environment with $PYTHON_CMD"
        exit 1
    }
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate" || {
    echo "ERROR: Failed to activate virtual environment"
    exit 1
}

# Verify we're using the virtual environment Python
echo "Python executable after activation: $(which python)"
echo "Python version after activation: $(python --version)"

# Install required packages in the virtual environment
echo "Installing required packages..."
# Upgrade pip within the venv first
python -m pip install --upgrade pip || echo "Warning: pip upgrade failed, continuing anyway"

# Install primary dependencies
echo "Installing primary dependencies..."
pip install --no-cache-dir networkx scipy tqdm joblib python-louvain matplotlib || {
    echo "ERROR: Failed to install primary dependencies"
    exit 1
}

# Install rnbrw (Simplified - assumes it's on PyPI or installed manually in the venv)
echo "Installing rnbrw..."
pip install --no-cache-dir rnbrw || {
    echo "ERROR: Failed to install rnbrw. Ensure it is available on PyPI or install manually into $VENV_DIR."
    # If 'pip install rnbrw' fails and you have the source code locally (e.g., in your home dir):
    # echo "Trying to install rnbrw from local source..."
    # pip install --no-cache-dir /path/to/your/rnbrw/source/directory
    # Uncomment and edit the line above if needed, otherwise delete it.
    exit 1
}

# Verify packages can be imported (Keep this check)
echo "Verifying package imports..."
python -c "
import sys
print('Python path:')
for p in sys.path:
    print(f'  {p}')

print('\\nChecking imports:')
dependencies = [
    'networkx',
    'scipy',
    'tqdm',
    'joblib',
    'matplotlib',
    'community',  # python-louvain package
    'rnbrw'
]

for pkg in dependencies:
    try:
        __import__(pkg)
        print(f'{pkg} ✓')
    except ImportError as e:
        print(f'{pkg} ✗ - {e}')
        sys.exit(1) # Exit if essential package failed to import

# Test rnbrw functionality (optional but good)
try:
    import rnbrw
    print('\\nTesting rnbrw.compute_weights function (basic check):')
    import networkx as nx
    # Use a small graph for quick test
    G_test = nx.path_graph(4)
    # Call compute_weights; adjust args if needed by your specific rnbrw implementation
    # Add dummy init_weight if required by your specific rnbrw fork
    weights = rnbrw.compute_weights(G_test, init_weight=0.0000001)
    if hasattr(weights, 'shape'):
         print(f'compute_weights returned shape: {weights.shape}')
    else:
         print(f'compute_weights returned type: {type(weights)}')
    print('rnbrw basic check successful.')
except Exception as e:
    print(f'\\nError testing rnbrw: {e}')
    # Decide if this should be fatal
    # sys.exit(1)
" || exit 1 # Exit Slurm script if Python verification fails



# Verify script exists
SCRIPT_PATH="script_path/run_rnbrw_batch.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi
echo "Python script found at: $SCRIPT_PATH"

# Run the Python script
echo "Starting Python multiprocessing script ($SLURM_CPUS_PER_TASK cores available) at $(date)..."
python "$SCRIPT_PATH" ${SLURM_ARRAY_TASK_ID}
PYTHON_EXIT_CODE=$?
echo "Python script finished at $(date) with exit code $PYTHON_EXIT_CODE"

# Check the exit code of the Python script
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "Error: Python script exited with non-zero status: $PYTHON_EXIT_CODE"
    
    # If it failed, let's get more diagnostic information
    echo "Running script with trace to see exact failure point..."
    python -m trace --trace "$SCRIPT_PATH" > script_trace.log 2>&1 || echo "Failed to run trace"
    
    echo "First 20 lines of trace log (full log in script_trace.log):"
    head -n 20 script_trace.log
    
    echo "
Script failed with error code $PYTHON_EXIT_CODE.
Possible issues:
1. Missing dependencies - try 'pip install -e <path_to_rnbrw>' in virtual env
2. Version incompatibilities - check if rnbrw is compatible with Python 3.6
3. Code issues in your script - review trace log for details
"
    exit $PYTHON_EXIT_CODE
else
    echo "Job completed successfully at $(date)"
fi

# Deactivate virtual environment
deactivate

