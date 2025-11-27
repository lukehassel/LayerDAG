
import sys
sys.path.append('mpo')

import torch
import numpy as np
import os
from tqdm import tqdm
from qiskit.converters import circuit_to_dag

# Import from mpo/circuit_utils.py
from circuit_utils import (
    create_random_circuit_with_universal_gates,  # Main function for circuit generation
    dag_to_gate_and_edges,
    reconstruct_circuit_from_noisy,
    get_universal_gate_set
)
from fidelity import get_fidelity
from circuit_cost import calculate_efficiency_score

# Import diffusion models
import importlib.util
spec = importlib.util.spec_from_file_location("diffusion", "src/model/diffusion.py")
diffusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diffusion)
DiscreteDiffusion = diffusion.DiscreteDiffusion
EdgeDiscreteDiffusion = diffusion.EdgeDiscreteDiffusion


DATASET_NAME = 'quantum_circuits'
NUM_TRAIN = 500
NUM_VAL = 100
NUM_TEST = 100

# Circuit generation parameters (for create_random_circuit_with_universal_gates)
MIN_QUBITS = 3
MAX_QUBITS = 6
MIN_DEPTH = 5
MAX_DEPTH = 15

# Diffusion parameters
T = 100  # Number of diffusion timesteps
TIMESTEP_RANGE = (0, 100)  # Range of timesteps to sample from
NOISE_TYPES = ['gate_only', 'edge_only', 'both']


def create_noisy_circuit_with_fidelity(num_qubits, depth, seed, timestep, noise_type='gate_only'):
    # Use create_random_circuit_with_universal_gates() from circuit_utils.py
    original_circuit = create_random_circuit_with_universal_gates(
        num_qubits=num_qubits,
        depth=depth,
        seed=seed,
        max_operands=2
    )

    # Convert to DAG representation
    dag = circuit_to_dag(original_circuit)
    gate_indices, adjacency, gate_to_idx, gate_info = dag_to_gate_and_edges(dag)

    # Handle timestep 0 (no noise)
    if timestep == 0:
        fidelity_result = get_fidelity(original_circuit, original_circuit)
        return gate_indices, adjacency, gate_to_idx, gate_info, fidelity_result['fidelity']

    # Initialize diffusion models with proper dimensions
    num_circuit_gate_types = len(gate_to_idx)
    marginal = torch.ones(num_circuit_gate_types) / num_circuit_gate_types

    gate_diffusion = DiscreteDiffusion(
        marginal_list=[marginal],
        T=T,
        s=0.008
    )

    avg_in_deg = adjacency.sum() / adjacency.shape[0] if adjacency.shape[0] > 0 else 1.0
    edge_diffusion = EdgeDiscreteDiffusion(
        avg_in_deg=avg_in_deg.item(),
        T=T,
        s=0.008
    )

    # Apply noise based on type
    t = torch.tensor([timestep])
    noisy_gate_indices = gate_indices
    noisy_adjacency = adjacency

    if noise_type == 'gate_only' or noise_type == 'both':
        _, noisy_gate_indices = gate_diffusion.apply_noise(gate_indices, t=t)

    if noise_type == 'edge_only' or noise_type == 'both':
        _, noisy_adj_flat = edge_diffusion.apply_noise(adjacency, t=t)
        noisy_adjacency = noisy_adj_flat.reshape(adjacency.shape)

    # Reconstruct noisy circuit
    idx_to_gate = {idx: gate for gate, idx in gate_to_idx.items()}
    noisy_circuit = reconstruct_circuit_from_noisy(
        noisy_gate_indices, noisy_adjacency, idx_to_gate, gate_info, num_qubits
    )

    # Calculate fidelity using get_fidelity() from fidelity.py
    fidelity_result = get_fidelity(original_circuit, noisy_circuit)

    return noisy_gate_indices, noisy_adjacency, gate_to_idx, gate_info, fidelity_result['fidelity']


def convert_to_layerdag_format(gate_indices, adjacency, gate_info, num_qubits, fidelity):
    n_gates = len(gate_indices)

    # Extract edges from adjacency matrix
    edge_list = []
    for i in range(n_gates):
        for j in range(n_gates):
            if adjacency[i, j] > 0.5:
                edge_list.append((j, i))  # (src, dst)

    if len(edge_list) == 0:
        # Create minimal valid graph
        src = torch.tensor([0], dtype=torch.long)
        dst = torch.tensor([0], dtype=torch.long)
    else:
        src = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edge_list], dtype=torch.long)

    # Node features: [gate_type, qubit_0, qubit_1]
    # Include wire information as described in the paper
    x_n = []
    for i in range(n_gates):
        gate_type_idx = gate_indices[i].item()
        _, qubits, _ = gate_info[i]

        # Extract qubit information and shift by +1 to handle -1 values
        # -1 (no qubit for single-qubit gates) becomes 0
        qubit_0 = (qubits[0] + 1) if len(qubits) > 0 else 0
        qubit_1 = (qubits[1] + 1) if len(qubits) > 1 else 0

        x_n.append([gate_type_idx, qubit_0, qubit_1])

    x_n = torch.tensor(x_n, dtype=torch.long)

    # Remap qubit columns to ensure consecutive integers starting from 0
    # This is required by the diffusion model's assertion that values must be
    # consecutive starting from 0. We remap both columns independently to
    # eliminate any gaps in the indices.
    if x_n.shape[0] > 0 and x_n.ndim == 2:
        # Remap qubit_0 column (column 1)
        unique_q0 = torch.unique(x_n[:, 1])
        q0_map = {old_val.item(): new_val for new_val, old_val in enumerate(unique_q0)}
        for i in range(x_n.shape[0]):
            x_n[i, 1] = q0_map[x_n[i, 1].item()]

        # Remap qubit_1 column (column 2)
        unique_q1 = torch.unique(x_n[:, 2])
        q1_map = {old_val.item(): new_val for new_val, old_val in enumerate(unique_q1)}
        for i in range(x_n.shape[0]):
            x_n[i, 2] = q1_map[x_n[i, 2].item()]

    # Calculate efficiency score based on gate count
    efficiency_score = calculate_efficiency_score(n_gates, num_qubits)

    # Multi-label target: [fidelity, efficiency_score]
    y = [float(fidelity), float(efficiency_score)]

    return src, dst, x_n, y


def generate_dataset(num_samples, seed_offset=0):
    dataset = {
        'src_list': [],
        'dst_list': [],
        'x_n_list': [],
        'y_list': []
    }

    print(f"Generating {num_samples} samples using create_random_circuit_with_universal_gates()...")

    for i in tqdm(range(num_samples)):
        # Random parameters
        num_qubits = np.random.randint(MIN_QUBITS, MAX_QUBITS + 1)
        depth = np.random.randint(MIN_DEPTH, MAX_DEPTH + 1)
        seed = seed_offset + i
        timestep = np.random.randint(TIMESTEP_RANGE[0], TIMESTEP_RANGE[1] + 1)
        noise_type = np.random.choice(NOISE_TYPES)

        try:
            # Generate circuit with noise and calculate fidelity
            gate_indices, adjacency, gate_to_idx, gate_info, fidelity = create_noisy_circuit_with_fidelity(
                num_qubits, depth, seed, timestep, noise_type
            )

            # Convert to LayerDAG format (includes wire information and multi-label)
            src, dst, x_n, y = convert_to_layerdag_format(gate_indices, adjacency, gate_info, num_qubits, fidelity)

            # Add to dataset
            dataset['src_list'].append(src)
            dataset['dst_list'].append(dst)
            dataset['x_n_list'].append(x_n)
            dataset['y_list'].append(y)

        except Exception as e:
            print(f"\nWarning: Failed to generate sample {i}: {e}")
            continue

    print(f"Successfully generated {len(dataset['y_list'])} samples")
    return dataset


def add_all_gates_circuit(dataset):
    from circuit_utils import create_circuit_with_all_gates

    # Create circuit with all gates
    circuit = create_circuit_with_all_gates()
    dag = circuit_to_dag(circuit)
    gate_indices, adjacency, gate_to_idx, gate_info = dag_to_gate_and_edges(dag)

    # Convert to LayerDAG format
    n_gates = len(gate_indices)
    x_n = []
    src = []
    dst = []

    # Build edge lists
    for i in range(n_gates):  # i = destination
        for j in range(n_gates):  # j = source
            if adjacency[i, j] > 0.5:
                src.append(j)
                dst.append(i)

    if len(src) == 0:
        src = torch.tensor([0], dtype=torch.long)
        dst = torch.tensor([0], dtype=torch.long)
    else:
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)

    # Build node features with wire information
    for i in range(n_gates):
        gate_type_idx = gate_indices[i].item()
        _, qubits, _ = gate_info[i]

        # Extract qubit information and shift by +1
        qubit_0 = (qubits[0] + 1) if len(qubits) > 0 else 0
        qubit_1 = (qubits[1] + 1) if len(qubits) > 1 else 0

        x_n.append([gate_type_idx, qubit_0, qubit_1])

    x_n = torch.tensor(x_n, dtype=torch.long)

    # Remap qubit columns to ensure consecutive integers
    if x_n.shape[0] > 0 and x_n.ndim == 2:
        # Remap qubit_0 column (column 1)
        unique_q0 = torch.unique(x_n[:, 1])
        q0_map = {old_val.item(): new_val for new_val, old_val in enumerate(unique_q0)}
        for i in range(x_n.shape[0]):
            x_n[i, 1] = q0_map[x_n[i, 1].item()]

        # Remap qubit_1 column (column 2)
        unique_q1 = torch.unique(x_n[:, 2])
        q1_map = {old_val.item(): new_val for new_val, old_val in enumerate(unique_q1)}
        for i in range(x_n.shape[0]):
            x_n[i, 2] = q1_map[x_n[i, 2].item()]

    # Dummy labels: fidelity=1.0 (perfect), efficiency=0.5 (reference)
    y = [1.0, 0.5]

    # Add to dataset
    dataset['src_list'].append(src)
    dataset['dst_list'].append(dst)
    dataset['x_n_list'].append(x_n)
    dataset['y_list'].append(y)

    # Verify all gates are present
    unique_gates = sorted(torch.unique(x_n[:, 0]).tolist())
    print(f"  Added all-gates circuit: {len(unique_gates)} unique gate types {unique_gates}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Quantum Circuit Dataset Generation for LayerDAG")
    print("Using create_random_circuit_with_universal_gates() from circuit_utils.py")
    print("Includes WIRE INFORMATION as described in the paper")
    print("=" * 70)
    print(f"\nDataset: {DATASET_NAME}")
    print(f"  Train: {NUM_TRAIN} samples")
    print(f"  Val: {NUM_VAL} samples")
    print(f"  Test: {NUM_TEST} samples")
    print(f"\nCircuit parameters:")
    print(f"  Qubits: {MIN_QUBITS}-{MAX_QUBITS}")
    print(f"  Depth: {MIN_DEPTH}-{MAX_DEPTH}")
    print(f"  Gate set: {len(get_universal_gate_set()['all'])} universal gates")
    print(f"\nNode features:")
    print(f"  [0] Gate type (18 universal gates)")
    print(f"  [1] Qubit 0 (wire information)")
    print(f"  [2] Qubit 1 (wire information, -1 for single-qubit gates)")
    print(f"\nDiffusion parameters:")
    print(f"  Timesteps: T={T}")
    print(f"  Timestep range: {TIMESTEP_RANGE}")
    print(f"  Noise types: {NOISE_TYPES}")
    print(f"\nLabels (multi-label):")
    print(f"  [0] Fidelity (0-1): similarity to original, from get_fidelity()")
    print(f"  [1] Efficiency (0-1): circuit quality, from circuit_cost.py")
    print(f"      - 0.5 = reference gate count (num_qubits * 10)")
    print(f"      - 1.0 = very efficient (few gates)")
    print(f"      - 0.0 = very inefficient (many gates)")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = f'data_files/{DATASET_NAME}_processed'
    os.makedirs(output_dir, exist_ok=True)

    # Generate datasets
    print("\n--- Train Set ---")
    train_set = generate_dataset(NUM_TRAIN, seed_offset=0)
    add_all_gates_circuit(train_set)

    print("\n--- Val Set ---")
    val_set = generate_dataset(NUM_VAL, seed_offset=10000)
    add_all_gates_circuit(val_set)

    print("\n--- Test Set ---")
    test_set = generate_dataset(NUM_TEST, seed_offset=20000)
    add_all_gates_circuit(test_set)

    # Save datasets
    torch.save(train_set, os.path.join(output_dir, 'train.pth'))
    torch.save(val_set, os.path.join(output_dir, 'val.pth'))
    torch.save(test_set, os.path.join(output_dir, 'test.pth'))

    print("\n" + "=" * 70)
    print("Dataset Generation Complete!")
    print("=" * 70)
    print(f"\nSaved to: {output_dir}/")
    print(f"  train.pth: {len(train_set['y_list'])} samples")
    print(f"  val.pth: {len(val_set['y_list'])} samples")
    print(f"  test.pth: {len(test_set['y_list'])} samples")

    # Statistics
    print(f"\nLabel statistics (multi-label: [fidelity, efficiency]):")
    for name, data in [('Train', train_set), ('Val', val_set), ('Test', test_set)]:
        y_array = np.array(data['y_list'])  # Shape: (n_samples, 2)
        fidelities = y_array[:, 0]
        efficiencies = y_array[:, 1]
        print(f"\n  {name} Set:")
        print(f"    Fidelity:   mean={np.mean(fidelities):.4f}, std={np.std(fidelities):.4f}, "
              f"min={np.min(fidelities):.4f}, max={np.max(fidelities):.4f}")
        print(f"    Efficiency: mean={np.mean(efficiencies):.4f}, std={np.std(efficiencies):.4f}, "
              f"min={np.min(efficiencies):.4f}, max={np.max(efficiencies):.4f}")

    all_x_n = torch.cat(train_set['x_n_list'] + val_set['x_n_list'] + test_set['x_n_list'])
    print(f"\nNode features (with wire information):")
    print(f"  Shape: {all_x_n.shape} (n_gates, 3)")
    print(f"  Feature 0 - Gate types: range {all_x_n[:, 0].min().item()} - {all_x_n[:, 0].max().item()}")
    print(f"  Feature 1 - Qubit 0: range {all_x_n[:, 1].min().item()} - {all_x_n[:, 1].max().item()}")
    print(f"  Feature 2 - Qubit 1: range {all_x_n[:, 2].min().item()} - {all_x_n[:, 2].max().item()} (-1 for single-qubit gates)")
    print(f"  Num gate type categories: {all_x_n[:, 0].max().item() + 1}")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Dataset includes:")
    print("     - Wire information (qubit connections)")
    print("     - Multi-label: [fidelity, efficiency_score]")
    print("  2. Run training:")
    print("     python train.py --config_file configs/LayerDAG/quantum_circuits.yaml")
    print("=" * 70)


if __name__ == "__main__":
    main()
