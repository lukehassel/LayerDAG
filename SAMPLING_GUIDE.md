# Quantum Circuits Sampling & Evaluation Guide

This guide explains how to sample quantum circuits from your trained LayerDAG model and evaluate the generated results.

## Overview

Three scripts have been created for you:

1. **`sample_quantum_circuits.py`** - Generates quantum circuits from the trained model
2. **`evaluate_quantum_circuits.py`** - Evaluates the quality of generated circuits
3. **`batch_scripts/run_sampling.sh`** - SLURM script for automated sampling and evaluation

---

## Quick Start (Recommended)

The easiest way to sample and evaluate is using the SLURM script:

```bash
# Submit the sampling job
sbatch batch_scripts/run_sampling.sh
```

This will automatically:
- Find your latest trained model (`model_quantum_circuits_*.pth`)
- Generate circuits for validation and test sets
- Evaluate the generated circuits
- Save all results with timestamp

**Output location:** `quantum_circuits_samples_YYYYMMDD_HHMMSS/`

---

## Manual Usage

### 1. Sampling Circuits

Generate circuits conditioned on validation/test set labels:

```bash
# Activate the virtual environment
source venv_training/bin/activate

# Run sampling
python sample_quantum_circuits.py \
    --model_path model_quantum_circuits_Nov18-19:54:47.pth \
    --output_dir my_samples \
    --batch_size 64 \
    --sample_val \
    --sample_test
```

**Key Arguments:**
- `--model_path`: Path to your trained model checkpoint (required)
- `--output_dir`: Where to save generated circuits (default: `quantum_circuits_samples`)
- `--batch_size`: Number of circuits to generate in parallel (default: 64)
- `--sample_val`: Generate validation set circuits (default: True)
- `--sample_test`: Generate test set circuits (default: True)
- `--sample_train`: Generate training set circuits (default: False)

**Advanced Sampling Options:**
```bash
# Control diffusion sampling steps for faster/higher quality generation
python sample_quantum_circuits.py \
    --model_path model_quantum_circuits_*.pth \
    --min_num_steps_n 32 \
    --max_num_steps_n 64 \
    --min_num_steps_e 8 \
    --max_num_steps_e 16
```

**Output Files:**
- `validation.pth` - Generated validation circuits
- `test.pth` - Generated test circuits
- `train.pth` - Generated training circuits (if requested)

---

### 2. Evaluating Generated Circuits

Evaluate the quality of generated circuits:

```bash
# Evaluate validation set
python evaluate_quantum_circuits.py \
    --generated_file my_samples/validation.pth \
    --dataset validation \
    --output_dir my_samples/validation_eval

# Evaluate test set
python evaluate_quantum_circuits.py \
    --generated_file my_samples/test.pth \
    --dataset test \
    --output_dir my_samples/test_eval
```

**Key Arguments:**
- `--generated_file`: Path to generated circuits .pth file (required)
- `--dataset`: Which real dataset to compare against (`train`, `validation`, or `test`)
- `--output_dir`: Where to save evaluation results

**Output Files:**
- `evaluation_plots.png` - Visualization comparing real vs generated
- `evaluation_report.txt` - Text report with metrics

---

## Understanding the Results

### Evaluation Metrics

The evaluation computes three categories of metrics:

#### 1. **Validity Metrics**
Checks if generated circuits are structurally valid:
- **Valid DAG structure**: No cycles in the circuit dependency graph
- **Valid gate types**: All gates are in range [0-17] (18 universal gates)
- **Valid qubit indices**: Qubit indices are consecutive starting from 0

**Goal:** 100% validity rate

#### 2. **Quality Metrics**
Compares generated circuits to real circuits:
- **Label Distribution**: How well generated circuits match target fidelity/efficiency
  - Fidelity MAE (Mean Absolute Error)
  - Efficiency MAE
- **Circuit Size**: Number of gates comparison
- **Edge Density**: Number of dependencies comparison

**Goal:** Low MAE, similar distributions

#### 3. **Diversity Metrics**
Measures variety in generated circuits:
- **Uniqueness**: Percentage of unique circuits generated
- **Gate Type Distribution**: Coverage of different gate types

**Goal:** High uniqueness (>90%), diverse gate usage

---

### Example Evaluation Report

```
==========================================
QUANTUM CIRCUITS EVALUATION REPORT
==========================================

VALIDITY METRICS
--------------------------------------------------
Total circuits: 101
Valid DAGs: 101 (100.0%)
Valid gates: 101 (100.0%)
Valid qubits: 101 (100.0%)
Fully valid: 101 (100.0%)

QUALITY METRICS
--------------------------------------------------
Fidelity MAE: 0.045
Efficiency MAE: 0.032
Size difference: 2.3 gates

DIVERSITY METRICS
--------------------------------------------------
Uniqueness: 98.0%
Unique circuits: 99 / 101
```

---

### Visualization Plots

The `evaluation_plots.png` contains 6 subplots:

1. **Fidelity Distribution**: Histogram of fidelity values (real vs generated)
2. **Efficiency Distribution**: Histogram of efficiency values
3. **Circuit Size Distribution**: Number of gates per circuit
4. **Fidelity Scatter**: Real vs Generated fidelity (should be close to diagonal)
5. **Efficiency Scatter**: Real vs Generated efficiency
6. **Edge Count Distribution**: Number of dependencies per circuit

**What to look for:**
- Distributions should overlap (real and generated)
- Scatter plots should cluster near the diagonal line
- Similar means and standard deviations

---

## Advanced Usage

### Custom Sampling Strategy

Generate circuits with specific diffusion parameters:

```bash
# Fast sampling (fewer steps, lower quality)
python sample_quantum_circuits.py \
    --model_path model_*.pth \
    --min_num_steps_n 16 \
    --max_num_steps_n 32 \
    --min_num_steps_e 4 \
    --max_num_steps_e 8

# High-quality sampling (more steps, slower)
python sample_quantum_circuits.py \
    --model_path model_*.pth \
    --min_num_steps_n 64 \
    --max_num_steps_n 64 \
    --min_num_steps_e 16 \
    --max_num_steps_e 16
```

### Batch Sampling Multiple Seeds

```bash
# Generate multiple samples with different seeds
for seed in 0 1 2 3 4; do
    python sample_quantum_circuits.py \
        --model_path model_*.pth \
        --output_dir samples_seed${seed} \
        --seed $seed
done
```

### Custom Evaluation

```python
import torch
from src.dataset import load_dataset

# Load generated circuits
data = torch.load('my_samples/validation.pth')

# Access individual circuits
for i in range(len(data['src_list'])):
    src = data['src_list'][i]
    dst = data['dst_list'][i]
    x_n = data['x_n_list'][i]  # Shape: (num_gates, 3)
    y = data['y_list'][i]      # Shape: (2,) [fidelity, efficiency]

    # Gate types are in x_n[:, 0]
    gate_types = x_n[:, 0]

    # Qubit indices are in x_n[:, 1] and x_n[:, 2]
    qubit_0 = x_n[:, 1]
    qubit_1 = x_n[:, 2]

    # Your custom analysis here...
```

---

## Troubleshooting

### Problem: "No model checkpoint found"

**Solution:** Make sure you have a trained model file:
```bash
ls -l model_quantum_circuits_*.pth
```

If missing, run training first:
```bash
sbatch batch_scripts/run_training.sh
```

### Problem: "CUDA out of memory"

**Solution:** Reduce batch size:
```bash
python sample_quantum_circuits.py \
    --model_path model_*.pth \
    --batch_size 32  # or 16
```

### Problem: Low validity rate (<100%)

**Possible causes:**
- Model not fully trained
- Dataset has issues
- Bug in sampling code

**Solution:** Check training logs and dataset statistics

### Problem: High MAE in labels

**Possible causes:**
- Conditional generation not working properly
- Model needs more training
- Labels are noisy

**Solution:**
1. Check if model was trained with `conditional: true`
2. Increase training epochs
3. Verify label quality in dataset

---

## File Structure

After running sampling and evaluation, you'll have:

```
quantum_circuits_samples_YYYYMMDD_HHMMSS/
├── validation.pth                    # Generated validation circuits
├── test.pth                          # Generated test circuits
├── validation_eval/
│   ├── evaluation_plots.png         # Validation visualizations
│   └── evaluation_report.txt        # Validation metrics
└── test_eval/
    ├── evaluation_plots.png         # Test visualizations
    └── evaluation_report.txt        # Test metrics
```

---

## Next Steps

After evaluating your model:

1. **If results look good:**
   - Use generated circuits for your downstream tasks
   - Share results/visualizations in your paper
   - Generate more samples with different seeds

2. **If validity is low:**
   - Check dataset for issues
   - Retrain model with more epochs
   - Debug edge prediction

3. **If quality is low (high MAE):**
   - Increase model capacity (more layers, larger embeddings)
   - Train longer
   - Check conditional generation setup

4. **If diversity is low:**
   - Check if model is mode-collapsing
   - Try different sampling strategies
   - Increase temperature during sampling

---

## Citation

If you use this code, please cite the LayerDAG paper:

```bibtex
@inproceedings{layerdag2023,
  title={LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graph Generation},
  author={...},
  booktitle={...},
  year={2023}
}
```

---

## Support

For issues or questions:
1. Check the logs in `logs/sampling_*.out`
2. Verify your trained model works by loading it manually
3. Check GitHub issues at the LayerDAG repository

Good luck with your quantum circuit generation!
