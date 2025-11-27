#!/bin/bash
#SBATCH --job-name=layerdag_train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=c23g
#SBATCH -A lect0163
#SBATCH --mail-user=luke.hassel@rwth-aachen.de
#SBATCH --mail-type=BEGIN,END,FAIL

# Create logs directory if it doesn't exist
mkdir -p logs

# Create timestamped log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/layerdag_${TIMESTAMP}_${SLURM_JOB_ID}.out"
ERR_FILE="logs/layerdag_${TIMESTAMP}_${SLURM_JOB_ID}.err"

# Redirect output to timestamped log files
exec 1> >(tee -a "${LOG_FILE}")
exec 2> >(tee -a "${ERR_FILE}" >&2)

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

cd /rwthfs/rz/cluster/home/wo057552/LayerDAG

# Create virtual environment if it doesn't exist
if [ ! -d "venv_training" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_training
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_training/bin/activate

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
CUDA_LAUNCH_BLOCKING=1 python train.py --config_file configs/LayerDAG/quantum_circuits.yaml

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed"
echo "=========================================="
