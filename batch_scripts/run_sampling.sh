#!/bin/bash
#SBATCH --job-name=layerdag_sample
#SBATCH --time=02:00:00
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
LOG_FILE="logs/sampling_${TIMESTAMP}_${SLURM_JOB_ID}.out"
ERR_FILE="logs/sampling_${TIMESTAMP}_${SLURM_JOB_ID}.err"

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

# Activate virtual environment (should already exist from training)
echo "Activating virtual environment..."
source venv_training/bin/activate

# Check if virtual environment exists
if [ ! -d "venv_training" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run the training script first to set up the environment."
    exit 1
fi

# Find the most recent model checkpoint
echo "Looking for model checkpoint..."
MODEL_PATH=$(ls -t model_quantum_circuits_*.pth 2>/dev/null | head -n 1)

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No model checkpoint found!"
    echo "Expected files matching: model_quantum_circuits_*.pth"
    exit 1
fi

echo "Found model: $MODEL_PATH"

# Create output directory with timestamp
OUTPUT_DIR="quantum_circuits_samples_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

echo ""
echo "=========================================="
echo "SAMPLING CONFIGURATION"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch size: 64"
echo "Sampling validation set: Yes"
echo "Sampling test set: Yes"
echo "Sampling training set: No"
echo "=========================================="
echo ""

# Run sampling
echo "Starting sampling..."
python sample_quantum_circuits.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 64 \
    --sample_val \
    --sample_test \
    --num_threads 8 \
    --seed 42

SAMPLING_EXIT_CODE=$?

if [ $SAMPLING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Sampling completed successfully!"

    # Run evaluation on validation set
    echo ""
    echo "=========================================="
    echo "RUNNING EVALUATION"
    echo "=========================================="
    echo ""

    # Evaluate validation set
    echo "Evaluating validation set..."
    python evaluate_quantum_circuits.py \
        --generated_file "${OUTPUT_DIR}/validation.pth" \
        --dataset validation \
        --output_dir "${OUTPUT_DIR}/validation_eval"

    VAL_EVAL_EXIT_CODE=$?

    # Evaluate test set
    echo ""
    echo "Evaluating test set..."
    python evaluate_quantum_circuits.py \
        --generated_file "${OUTPUT_DIR}/test.pth" \
        --dataset test \
        --output_dir "${OUTPUT_DIR}/test_eval"

    TEST_EVAL_EXIT_CODE=$?

    if [ $VAL_EVAL_EXIT_CODE -eq 0 ] && [ $TEST_EVAL_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Evaluation completed successfully!"
        echo ""
        echo "=========================================="
        echo "RESULTS SUMMARY"
        echo "=========================================="
        echo "Generated circuits saved to:"
        echo "  - ${OUTPUT_DIR}/validation.pth"
        echo "  - ${OUTPUT_DIR}/test.pth"
        echo ""
        echo "Evaluation results saved to:"
        echo "  - ${OUTPUT_DIR}/validation_eval/"
        echo "  - ${OUTPUT_DIR}/test_eval/"
        echo ""
        echo "View evaluation plots:"
        echo "  - ${OUTPUT_DIR}/validation_eval/evaluation_plots.png"
        echo "  - ${OUTPUT_DIR}/test_eval/evaluation_plots.png"
        echo ""
        echo "View evaluation reports:"
        echo "  - ${OUTPUT_DIR}/validation_eval/evaluation_report.txt"
        echo "  - ${OUTPUT_DIR}/test_eval/evaluation_report.txt"
        echo "=========================================="
    else
        echo ""
        echo "⚠ Warning: Evaluation failed!"
        echo "Validation eval exit code: $VAL_EVAL_EXIT_CODE"
        echo "Test eval exit code: $TEST_EVAL_EXIT_CODE"
    fi
else
    echo ""
    echo "✗ Sampling failed with exit code: $SAMPLING_EXIT_CODE"
fi

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed"
echo "=========================================="

exit $SAMPLING_EXIT_CODE
