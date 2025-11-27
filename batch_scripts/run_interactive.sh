#!/bin/bash
# Run training interactively on a compute node

echo "Requesting interactive GPU node..."
echo "This may take a few moments..."
echo ""

srun --job-name=layerdag_interactive \
     --time=24:00:00 \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=16 \
     --mem=4G \
     --gres=gpu:1 \
     --partition=c23g \
     -A lect0163 \
     --pty bash -c "
         cd /rwthfs/rz/cluster/home/wo057552/LayerDAG

         module purge
        module load GCCcore/12.2.0 Python/3.10.8
        module load CUDA/11.8.0

        echo "Python 3.10.8 and CUDA 11.8.0 loaded successfully"
        python3 --version

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
        python train.py --config_file configs/LayerDAG/tpu_tile_test.yaml

        echo ""
        echo "=========================================="
        echo "End Time: $(date)"
        echo "Job completed"
        echo "=========================================="

     "
