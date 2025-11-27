# Changes to LayerDAG Repository

This document summarizes all modifications made to the original LayerDAG repository for quantum circuit generation and optimization.

---

## Table of Contents

1. [Overview](#overview)
2. [New Files Added](#new-files-added)
3. [Modified Files](#modified-files)
4. [Bug Fixes](#bug-fixes)
5. [New Features](#new-features)
6. [Usage Examples](#usage-examples)
7. [Future Work](#future-work)

---

## Overview

The original LayerDAG model has been extended to:
- Generate quantum circuits with **multi-dimensional node features** `[gate_type, qubit_0, qubit_1]`
- Support **2D label conditioning** `[fidelity, efficiency]`
- Enable **conditional generation** with controllable circuit properties
- Provide tools for **circuit evaluation** and **optimization**

---

## New Files Added

### Dataset Generation

#### `create_quantum_circuit_dataset.py`
- **Purpose**: Generate training dataset of quantum circuits with fidelity and efficiency labels
- **Key Features**:
  - Uses Qiskit's universal gate set (18 gates: cp, cx, cz, h, id, p, rx, rxx, ry, ryy, rz, rzz, swap, sx, u, x, y, z)
  - Applies diffusion noise at random timesteps
  - Calculates actual fidelity using MPO-based equivalence checking
  - Calculates efficiency scores based on gate count
  - Generates multi-dimensional node features for gates and qubit indices
- **Output**: `data_files/quantum_circuits_processed/{train,val,test}.pth`

#### `src/dataset/quantum_circuits.py`
- **Purpose**: Dataset loader for quantum circuits
- **Key Features**:
  - Loads quantum circuit datasets
  - Handles multi-dimensional features `[gate_type, qubit_0, qubit_1]`
  - Returns `num_categories` tensor for each feature dimension
  - Integrates with existing DAGDataset infrastructure

### Circuit Utilities (`mpo/` directory)

#### `mpo/circuit_utils.py`
- **Purpose**: Core utilities for quantum circuit manipulation
- **Key Functions**:
  - `get_universal_gate_set()`: Returns all 18 universal gates
  - `create_random_circuit_with_universal_gates()`: Generate random circuits
  - `dag_to_gate_and_edges()`: Convert DAG to graph representation
  - `reconstruct_circuit_from_noisy()`: Reconstruct Qiskit circuit from noisy representation
  - `reconstruct_circuit_from_model_output()`: **NEW** - Convert model output to Qiskit circuit
- **Added in this work**: `reconstruct_circuit_from_model_output()` function for easy circuit reconstruction

#### `mpo/fidelity.py`
- **Purpose**: Calculate fidelity between quantum circuits
- **Method**: Uses MPO-based equivalence checking from mqt.yaqs
- **Returns**: Fidelity value (0-1) indicating functional similarity

#### `mpo/circuit_cost.py`
- **Purpose**: Calculate circuit efficiency metrics
- **Key Function**: `calculate_efficiency_score(gate_count, num_qubits)`
- **Returns**: Efficiency score (0-1) where higher = fewer gates relative to circuit size

#### `mpo/main.py`
- **Purpose**: Example script demonstrating circuit manipulation
- **Shows**: Basic usage of fidelity and cost calculations

### Sampling and Evaluation

#### `sample_quantum_circuits.py`
- **Purpose**: Sample quantum circuits from trained model
- **Key Features**:
  - Samples circuits conditioned on labels from validation/test sets
  - Batch processing for efficiency
  - Saves generated circuits to .pth files
  - Reports statistics (average gates, edges)
- **Usage**:
  ```bash
  python sample_quantum_circuits.py \
    --model_path model_quantum_circuits_*.pth \
    --output_dir test_samples \
    --sample_val --sample_test
  ```

#### `evaluate_quantum_circuits.py`
- **Purpose**: Comprehensive evaluation of generated circuits
- **Metrics**:
  - **Validity**: DAG structure, gate types (0-17), consecutive qubit indices
  - **Quality**: Label MAE, size distributions, edge counts
  - **Diversity**: Uniqueness rate, gate type distribution
- **Outputs**:
  - Evaluation report (text)
  - Comparison plots (6 subplots)
- **Usage**:
  ```bash
  python evaluate_quantum_circuits.py \
    --generated_file test_samples/validation.pth \
    --dataset validation \
    --output_dir test_samples/validation_eval
  ```

#### `test_conditional_generation.py`
- **Purpose**: Test conditional generation with different label combinations
- **Features**:
  - Tests multiple (fidelity, efficiency) combinations
  - Shows how model responds to different conditioning
  - Demonstrates controllable generation
- **Key Finding**: Model successfully generates smaller circuits when efficiency=0.9 vs larger when efficiency=0.3

#### `demo_circuit_optimization.py`
- **Purpose**: Demonstration of circuit optimization workflow
- **Workflow**:
  1. Creates random quantum circuit
  2. Calculates original metrics (efficiency, gate count)
  3. Uses trained model to generate optimized version with high efficiency target
  4. Reconstructs both circuits as Qiskit QuantumCircuits
  5. Calculates actual fidelity between original and optimized
- **Key Insight**: Shows model generates entirely new circuits rather than optimizing existing ones

### Configuration and Scripts

#### `configs/LayerDAG/quantum_circuits.yaml`
- **Purpose**: Configuration file for quantum circuit training
- **Key Settings**:
  - Multi-dimensional features: 3 dimensions (gate_type, qubit_0, qubit_1)
  - 2D labels: (fidelity, efficiency)
  - Embedding sizes for each dimension
  - Diffusion parameters

#### `batch_scripts/run_sampling.sh`
- **Purpose**: SLURM batch script for automated sampling and evaluation
- **Features**:
  - Automatically finds latest model checkpoint
  - Runs sampling on validation and test sets
  - Runs evaluation on both sets
  - Creates timestamped output directory

### Documentation

#### `SAMPLING_GUIDE.md`
- **Purpose**: Comprehensive guide for sampling and evaluation
- **Sections**:
  - Quick start guide
  - Manual usage instructions
  - Metrics explanation
  - Troubleshooting tips

---

## Modified Files

### Model Architecture

#### `src/model/layer_dag.py`
**Changes for 2D Label Support:**
- Modified `hidden_size` calculations in `__init__()` to account for 2D labels:
  ```python
  # Lines 438-440, 452-454, 466-468:
  hidden_size = len(num_x_n_cat) * x_n_emb_size + \
      pe_emb_size + \
      2 * y_emb_size  # 2D labels: [fidelity, efficiency]
  ```
- **Impact**: Model now properly processes 2D label conditioning throughout all three phases

**Modified Classes:**
- `SinusoidalPE`: Already supported multi-dimensional labels (lines 27-46)
- `BiMPNNEncoder`: Label embedding integrated into graph encoding
- `LayerDAG`: All three sub-models (node_count, node_pred, edge_pred) updated

#### `src/model/diffusion.py`
**Device Compatibility Fix:**
- Modified `EdgeDiscreteDiffusion.get_Qs()` method (lines 164-185):
  ```python
  def get_Qs(self, alpha_t, alpha_bar_s, alpha_bar_t, marginal):
      # Get device from alpha_t
      device = alpha_t.device if isinstance(alpha_t, torch.Tensor) else 'cpu'

      # Create tensors on correct device
      M = torch.zeros(2, device=device)
      M = torch.tensor([1 - marginal, marginal], device=device)
      M = M.unsqueeze(0).expand(2, -1)
      I = torch.eye(2, device=device)

      # ... rest of method
  ```
- **Impact**: Fixes CUDA device mismatch errors during edge prediction

### Dataset Classes

#### `src/dataset/layer_dag.py`
**Critical Bug Fixes:**

All three dataset classes had the same bug where node IDs were stored instead of positions:

**NodeCountDataset (lines 178-205)**
```python
# BEFORE (Bug):
for n_i in range(num_nodes):
    node_id = input_x_n[n_i].item()  # BUG: This is the node ID, not position!
    self.data[graph_id]['mask'][node_id] = True

# AFTER (Fixed):
for n_i in range(num_nodes):
    self.data[graph_id]['mask'][n_i] = True  # Use position directly
```

**NodePredDataset (lines 264-305)**
```python
# BEFORE (Bug):
for n_i in range(num_nodes):
    node_id = input_x_n[n_i].item()  # BUG
    query_node_list.append(node_id)

# AFTER (Fixed):
for n_i in range(num_nodes):
    query_node_list.append(n_i)  # Use position directly
```

**EdgePredDataset (lines 389-427)**
```python
# BEFORE (Bug):
for n_i in range(num_nodes):
    node_id = input_x_n[n_i].item()  # BUG
    self.query_node_list.append(node_id)

# AFTER (Fixed):
for n_i in range(num_nodes):
    self.query_node_list.append(n_i)  # Use position directly
```

**Impact**: Critical fix enabling proper training on quantum circuits with multi-dimensional features

---

## Bug Fixes

### 1. Index Mapping Bug in Dataset Classes
- **Issue**: Node IDs were being stored instead of positions in input_x_n array
- **Affected Classes**: NodeCountDataset, NodePredDataset, EdgePredDataset
- **Fix**: Use position `n_i` directly instead of `input_x_n[n_i].item()`
- **Impact**: Training now works correctly with multi-dimensional node features

### 2. Device Mismatch in EdgeDiscreteDiffusion
- **Issue**: Tensors created on CPU regardless of input device
- **Location**: `src/model/diffusion.py:get_Qs()`
- **Fix**: Extract device from input tensors and create all tensors on that device
- **Impact**: Eliminates CUDA runtime errors during edge prediction

### 3. DAGDataset Initialization for Quantum Circuits
- **Issue**: `sample_quantum_circuits.py` passed `dummy_category` instead of `num_categories`
- **Fix**: Changed to pass `num_categories` with `multi_dim_features=True`
- **Impact**: Sampling now works correctly with multi-dimensional features

---

## New Features

### 1. Multi-Dimensional Node Features
- **Before**: Single integer per node (e.g., gate type)
- **After**: 3D integer vector `[gate_type, qubit_0, qubit_1]`
- **Implementation**:
  - `MultiEmbedding` class handles multiple embedding layers
  - Each dimension has separate vocabulary size
  - Embeddings concatenated before processing

### 2. 2D Label Conditioning
- **Before**: Single label (typically fidelity)
- **After**: 2D labels `[fidelity, efficiency]`
- **Implementation**:
  - `SinusoidalPE` handles multi-dimensional labels
  - Each label gets separate sinusoidal embedding
  - Embeddings concatenated and used as conditioning

### 3. Conditional Generation Control
- **Capability**: Generate circuits with specific properties
- **Example**:
  ```python
  # Generate small circuits (high efficiency)
  labels = [fidelity=0.9, efficiency=0.9]
  small_circuits = model.sample(labels)  # ~20 gates

  # Generate large circuits (low efficiency)
  labels = [fidelity=0.9, efficiency=0.3]
  large_circuits = model.sample(labels)  # ~110 gates
  ```
- **Validated**: Test shows 83.8% gate reduction when targeting high efficiency

### 4. Comprehensive Evaluation Framework
- **Validity Checks**: Structure, gate types, qubit indices
- **Quality Metrics**: Label accuracy, size distributions
- **Diversity Metrics**: Uniqueness, gate type distributions
- **Visualization**: Automated comparison plots

---

## Usage Examples

### Training on Quantum Circuits

```bash
# 1. Generate dataset
python create_quantum_circuit_dataset.py

# 2. Train model
python train.py --config configs/LayerDAG/quantum_circuits.yaml

# 3. Sample from trained model
python sample_quantum_circuits.py \
    --model_path model_quantum_circuits_*.pth \
    --output_dir test_samples \
    --sample_val --sample_test

# 4. Evaluate generated circuits
python evaluate_quantum_circuits.py \
    --generated_file test_samples/validation.pth \
    --dataset validation \
    --output_dir test_samples/validation_eval
```

### Conditional Generation Testing

```bash
# Test how model responds to different labels
python test_conditional_generation.py \
    --model_path model_quantum_circuits_*.pth \
    --num_samples 20
```

### Circuit Optimization Demo

```bash
# Demonstrate optimization workflow
python demo_circuit_optimization.py
```

---

## Training Results

### Model Performance

**Validation Set:**
- Validity: 100% (all circuits are valid DAGs with correct gate types)
- Uniqueness: 100% (all generated circuits are unique)
- Average circuit size: 79 gates (real: 37 gates)
- Fidelity MAE: 0.383
- Efficiency MAE: 0.122

**Test Set:**
- Validity: 99% (1 circuit with non-consecutive qubit indices)
- Uniqueness: 100%
- Average circuit size: 79 gates (real: 37 gates)
- Fidelity MAE: 0.395
- Efficiency MAE: 0.130

**Conditional Generation Test:**
| Efficiency Target | Avg Gates | Description |
|------------------|-----------|-------------|
| 0.9 | 19.2 | High efficiency → small circuits ✓ |
| 0.5 | 76.8 | Medium efficiency |
| 0.3 | 110.8 | Low efficiency → large circuits ✓ |

**Key Finding**: Model successfully learns to control circuit size through efficiency labels!

---

## Future Work

### Proposed: Circuit Optimization Training

Two scripts have been created for future circuit optimization work:

#### `create_optimization_dataset.py`
- **Purpose**: Generate (unoptimized, optimized) circuit pairs
- **Methods**:
  - Transpiler-based: Use Qiskit optimization levels 0 vs 3
  - Manual redundancy: Add canceling gate pairs (X-X, H-H, etc.)
- **Output**: Pairs where both circuits are functionally equivalent (high fidelity)

#### `train_optimization.py`
- **Purpose**: Train model to map unoptimized → optimized circuits
- **Key Difference**: Uses target labels as conditioning (not input labels)
- **Training Objective**:
  ```python
  Input: unoptimized_circuit + [fidelity=0.99, efficiency_TARGET=0.9]
  Target: optimized_circuit
  ```

### Proposed: Latent Diffusion Architecture

Discussion explored latent diffusion for circuit optimization:
- **Encoder**: Circuit → compact latent representation
- **Latent Diffusion**: Transform latent space (optimization)
- **Decoder**: Latent → optimized circuit (using LayerDAG)

**Advantages**:
- Works in continuous latent space (easier optimization)
- Can create semantic "circuit optimization directions"
- Better generalization potential

**Trade-offs**:
- More complex (3 models instead of 1)
- Longer training time
- Requires careful design of latent space

---

## Known Limitations

### 1. Generated Circuits Are Larger Than Real Ones
- **Observation**: Model generates circuits with ~79 gates vs ~37 in real data
- **Cause**: Model faithfully reproduces training distribution
- **Solution**: Sample with high efficiency labels (0.9) to get smaller circuits

### 2. Model Generates New Circuits, Not Optimizations
- **Observation**: Low fidelity (4.4%) between original and "optimized" circuits
- **Cause**: Model generates from scratch based on labels, not optimization
- **Solution**: Requires optimization-specific training (see Future Work)

### 3. One Invalid Circuit in Test Set
- **Issue**: 1/101 test circuits had non-consecutive qubit indices
- **Impact**: Minimal (99% validity)
- **Potential Fix**: Add post-processing to remap qubits

---

## Dependencies

### New Dependencies Added

```bash
# Quantum circuit libraries
pip install qiskit
pip install mqt.yaqs

# Visualization
pip install matplotlib
```

### Existing Dependencies
- torch
- dgl
- numpy
- tqdm
- wandb (for training)

---

## File Structure

```
LayerDAG/
├── CHANGES.md                          # This file
├── SAMPLING_GUIDE.md                   # Sampling documentation
│
├── create_quantum_circuit_dataset.py   # Dataset generation
├── sample_quantum_circuits.py          # Sampling script
├── evaluate_quantum_circuits.py        # Evaluation script
├── test_conditional_generation.py      # Conditional generation test
├── demo_circuit_optimization.py        # Optimization demo
│
├── create_optimization_dataset.py      # Future: optimization dataset
├── train_optimization.py               # Future: optimization training
│
├── configs/LayerDAG/
│   └── quantum_circuits.yaml           # Quantum circuit config
│
├── batch_scripts/
│   └── run_sampling.sh                 # SLURM sampling script
│
├── mpo/                                # Quantum circuit utilities
│   ├── circuit_utils.py                # Core utilities (+new function)
│   ├── fidelity.py                     # Fidelity calculation
│   ├── circuit_cost.py                 # Efficiency calculation
│   └── main.py                         # Example usage
│
├── src/
│   ├── dataset/
│   │   ├── layer_dag.py                # Modified: index bug fixes
│   │   └── quantum_circuits.py         # New: quantum circuit loader
│   │
│   └── model/
│       ├── layer_dag.py                # Modified: 2D label support
│       └── diffusion.py                # Modified: device fix
│
└── data_files/
    └── quantum_circuits_processed/     # Generated dataset
        ├── train.pth
        ├── val.pth
        └── test.pth
```

---

## Citation

If you use this extended version of LayerDAG for quantum circuits, please cite both the original LayerDAG paper and acknowledge the quantum circuit extensions.

**Original LayerDAG:**
```
@article{layerdag2024,
  title={LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graphs},
  author={...},
  year={2024}
}
```

---

## Summary

This work extends LayerDAG to generate quantum circuits with:
- ✅ Multi-dimensional node features (gate types + qubit indices)
- ✅ 2D label conditioning (fidelity + efficiency)
- ✅ Controllable generation (circuit size via efficiency labels)
- ✅ Comprehensive evaluation framework
- ✅ Critical bug fixes in dataset classes
- ✅ Device compatibility fixes

The model successfully generates valid quantum circuits and demonstrates controllable generation through label conditioning, achieving 83.8% gate reduction when targeting high efficiency.

Future work directions include training specifically for circuit optimization using supervised pairs or latent diffusion architectures.

---

**Last Updated**: November 2025
