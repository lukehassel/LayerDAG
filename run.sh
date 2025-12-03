#!/bin/bash
#SBATCH --job-name=layerdag_train
#SBATCH --output=logs/layerdag_%j.out
#SBATCH --error=logs/layerdag_%j.err
#SBATCH --time=200:00:00
#SBATCH --account=lect0163
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=c23g

# Ensure we run from the repository root (directory containing this script)
cd "$(dirname "$0")"

# Load Python 3.10.8 module and set up virtual environment

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

module purge
module load GCCcore/12.2.0 Python/3.10.8
module load CUDA/11.8.0

echo "Python 3.10.8 and CUDA 11.8.0 loaded successfully"
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# pip install -r requirements.txt

echo "Setup complete!"

# Run script
echo "Running ..."
#python train.py --config_file configs/LayerDAG/tpu_tile.yaml
#python train.py --config_file configs/LayerDAG/tpu_tile_test.yaml

# Use module mode so that the repo root is on PYTHONPATH and `src` can be imported
python3 -m encoder.dataset

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed"
echo "=========================================="
