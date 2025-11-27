#!/bin/bash
# Interactive dataset generation using srun

echo "Requesting interactive compute node..."
echo "This may take a few moments..."
echo ""

srun --job-name=qc_dataset_gen \
     --time=02:00:00 \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --mem=32G \
     -A lect0163 \
     --pty bash -c "
         cd /rwthfs/rz/cluster/home/wo057552/LayerDAG

         echo '=========================================='
         echo 'Interactive job started'
         echo 'Node: \$(hostname)'
         echo 'Start Time: \$(date)'
         echo '=========================================='
         echo ''

         # Load modules
         module purge
         module load GCCcore/12.2.0 Python/3.10.8 CUDA/11.8.0

         echo 'Modules loaded'
         python3 --version
         echo ''

         # Activate virtual environment
         source venv/bin/activate
         echo 'Virtual environment activated'
         echo ''

         # Verify packages
         echo 'Checking installed packages...'
         python -c \"import torch; print(f'PyTorch: {torch.__version__}')\"
         python -c \"import qiskit; print(f'Qiskit: {qiskit.__version__}')\"
         python -c \"import mqt.yaqs; print('mqt.yaqs: OK')\"
         echo ''

         # Run dataset generation
         echo '=========================================='
         echo 'Starting Quantum Circuit Dataset Generation'
         echo '=========================================='
         echo ''

         python create_quantum_circuit_dataset.py

         echo ''
         echo '=========================================='
         echo 'End Time: \$(date)'
         echo 'Dataset generation completed'
         echo '=========================================='
     "
