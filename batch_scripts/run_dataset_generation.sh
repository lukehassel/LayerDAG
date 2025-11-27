#!/bin/bash
#SBATCH --job-name=qc_dataset_gen
#SBATCH --output=logs/dataset_gen_%j.out
#SBATCH --error=logs/dataset_gen_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition=c23g

# Create logs directory if it doesn't exist
cd /rwthfs/rz/cluster/home/wo057552/LayerDAG
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

# Load required modules
module purge
module load GCCcore/12.2.0 Python/3.10.8
module load CUDA/11.8.0

echo "Modules loaded successfully"
python3 --version
echo ""

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment 'venv' not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install PyTorch 2.0 for H100 support if not already installed
echo "Checking PyTorch installation..."
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch 2.0.1+cu118..."
    pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
fi

# Install DGL if not already installed
if ! python -c "import dgl" 2>/dev/null; then
    echo "Installing DGL 1.1.2+cu118..."
    pip install dgl==1.1.2+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
fi

# Install other required packages
echo "Installing additional dependencies..."
pip install -q tqdm einops numpy==1.26.3

# Verify required packages
echo "Checking installed packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import qiskit; print(f'Qiskit: {qiskit.__version__}')"
python -c "import mqt.yaqs; print('mqt.yaqs: OK')"
echo ""

# Test qubit usage before full generation
echo "=========================================="
echo "Testing Qubit Usage"
echo "=========================================="
echo ""
python test_qubit_usage.py
if [ $? -ne 0 ]; then
    echo "ERROR: Qubit usage test failed!"
    exit 1
fi
echo ""

# Run dataset generation
echo "=========================================="
echo "Starting Quantum Circuit Dataset Generation"
echo "=========================================="
echo ""

python create_quantum_circuit_dataset.py

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Dataset generation completed"
echo "=========================================="
