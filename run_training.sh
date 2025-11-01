#!/bin/bash
#SBATCH --job-name=layerdag_train
#SBATCH --output=logs/layerdag_%j.out
#SBATCH --error=logs/layerdag_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=c23g

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

# Install PyTorch 2.0 for H100 support
echo "Installing PyTorch 2.0.1+cu118..."
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install DGL compatible with PyTorch 2.0
echo "Installing DGL 1.1.2+cu118..."
pip install dgl==1.1.2+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html

# Install other dependencies
echo "Installing additional dependencies..."
pip install tqdm einops wandb pydantic pandas

# Install specific numpy version
echo "Installing numpy 1.26.3..."
pip install numpy==1.26.3

echo "Setup complete!"

# Run training
echo "Starting training..."
#python train.py --config_file configs/LayerDAG/tpu_tile.yaml
python train.py --config_file configs/LayerDAG/tpu_tile_test.yaml

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed"
echo "=========================================="
