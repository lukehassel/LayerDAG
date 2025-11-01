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
     --pty bash -c "
         cd /rwthfs/rz/cluster/home/wo057552/LayerDAG

         echo '=========================================='
         echo 'Interactive job started'
         echo 'Node: \$(hostname)'
         echo 'Start Time: \$(date)'
         echo '=========================================='
         echo ''

         module purge
         module load GCCcore/12.2.0 Python/3.10.8 CUDA/11.8.0

         echo 'Modules loaded'
         python3 --version
         nvidia-smi
         echo ''

         source venv/bin/activate
         echo 'Virtual environment activated'
         echo ''

         echo 'Starting training...'
         python train.py --config_file configs/LayerDAG/random_small.yaml

         echo ''
         echo '=========================================='
         echo 'End Time: \$(date)'
         echo 'Interactive job completed'
         echo '=========================================='
     "
