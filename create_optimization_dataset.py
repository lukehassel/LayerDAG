"""
Create dataset for training circuit optimization.

Key idea: Instead of random noise, we create pairs of:
  - Unoptimized circuits (large, redundant)
  - Optimized circuits (small, efficient)
Both are functionally equivalent (high fidelity).
"""

import sys
sys.path.append('mpo')

import torch
import numpy as np
import os
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag

from circuit_utils import (
    create_random_circuit_with_universal_gates,
    dag_to_gate_and_edges,
    get_universal_gate_set
)
from fidelity import get_fidelity
from circuit_cost import calculate_efficiency_score


def create_optimization_pair_using_transpiler(num_qubits, depth, seed):
    """
    Create (unoptimized, optimized) pair using Qiskit transpiler.

    Strategy:
    1. Generate base circuit
    2. Unoptimized = transpile with optimization_level=0
    3. Optimized = transpile with optimization_level=3

    Returns:
        (unopt_circuit, opt_circuit, fidelity, unopt_eff, opt_eff)
    """
    # Create base circuit
    base = create_random_circuit_with_universal_gates(num_qubits, depth, seed)
    gates = get_universal_gate_set()['all']

    # Create unoptimized version (no optimization)
    unopt = transpile(base, basis_gates=gates, optimization_level=0)

    # Create optimized version (maximum optimization)
    opt = transpile(base, basis_gates=gates, optimization_level=3)

    # Only keep if optimization actually reduced gates
    if len(opt.data) >= len(unopt.data):
        return None

    # Measure fidelity (should be high - functionally equivalent)
    fidelity_result = get_fidelity(unopt, opt)
    fidelity = fidelity_result['fidelity']

    # Only keep high-fidelity pairs (functionally equivalent)
    if fidelity < 0.95:
        return None

    # Calculate efficiency
    unopt_eff = calculate_efficiency_score(len(unopt.data), unopt.num_qubits)
    opt_eff = calculate_efficiency_score(len(opt.data), opt.num_qubits)

    return unopt, opt, fidelity, unopt_eff, opt_eff


def add_redundant_gates(circuit, num_redundant):
    """
    Add redundant gate sequences that cancel out.

    Strategies:
    - X-X pairs (cancel)
    - H-H pairs (cancel)
    - Z-Z pairs (cancel)
    - CNOT-CNOT on same qubits (cancel)
    """
    from qiskit import QuantumCircuit

    # Create copy
    redundant = circuit.copy()
    num_qubits = circuit.num_qubits

    for _ in range(num_redundant):
        q = np.random.randint(0, num_qubits)
        gate_type = np.random.choice(['x', 'y', 'z', 'h'])

        # Add gate twice (cancels out)
        getattr(redundant, gate_type)(q)
        getattr(redundant, gate_type)(q)

    return redundant


def create_optimization_pair_manual(num_qubits, depth, seed):
    """
    Manually create (unoptimized, optimized) by adding redundancy.

    Returns:
        (unopt_circuit, opt_circuit, fidelity, unopt_eff, opt_eff)
    """
    # Create optimized circuit (the target)
    opt = create_random_circuit_with_universal_gates(num_qubits, depth, seed)

    # Create unoptimized by adding redundancy
    num_redundant = np.random.randint(5, 15)
    unopt = add_redundant_gates(opt, num_redundant)

    # Transpile both to ensure same basis
    gates = get_universal_gate_set()['all']
    opt = transpile(opt, basis_gates=gates, optimization_level=3)
    unopt = transpile(unopt, basis_gates=gates, optimization_level=0)

    # Must have actually added gates
    if len(unopt.data) <= len(opt.data):
        return None

    # Measure fidelity
    fidelity_result = get_fidelity(unopt, opt)
    fidelity = fidelity_result['fidelity']

    # Only keep high-fidelity pairs
    if fidelity < 0.95:
        return None

    # Calculate efficiency
    unopt_eff = calculate_efficiency_score(len(unopt.data), unopt.num_qubits)
    opt_eff = calculate_efficiency_score(len(opt.data), opt.num_qubits)

    return unopt, opt, fidelity, unopt_eff, opt_eff


def circuit_to_layerdag_format(circuit, fidelity, efficiency):
    """Convert Qiskit circuit to LayerDAG format."""
    dag = circuit_to_dag(circuit)
    gate_indices, adjacency, gate_to_idx, gate_info = dag_to_gate_and_edges(dag)

    n_gates = len(gate_indices)

    # Extract edges
    edge_list = []
    for i in range(n_gates):
        for j in range(n_gates):
            if adjacency[i, j] > 0.5:
                edge_list.append((j, i))

    if len(edge_list) == 0:
        src = torch.tensor([0], dtype=torch.long)
        dst = torch.tensor([0], dtype=torch.long)
    else:
        src = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edge_list], dtype=torch.long)

    # Node features: [gate_type, qubit_0, qubit_1]
    all_gates = sorted(get_universal_gate_set()['all'])
    x_n = []

    for i in range(n_gates):
        gate_name, qubits, _ = gate_info[i]
        gate_type_idx = all_gates.index(gate_name)

        qubit_0 = (qubits[0] + 1) if len(qubits) > 0 else 0
        qubit_1 = (qubits[1] + 1) if len(qubits) > 1 else 0

        x_n.append([gate_type_idx, qubit_0, qubit_1])

    x_n = torch.tensor(x_n, dtype=torch.long)

    # Remap qubits to consecutive integers
    if x_n.shape[0] > 0 and x_n.ndim == 2:
        # Remap qubit_0
        unique_q0 = torch.unique(x_n[:, 1])
        q0_map = {old_val.item(): new_val for new_val, old_val in enumerate(unique_q0)}
        for i in range(x_n.shape[0]):
            x_n[i, 1] = q0_map[x_n[i, 1].item()]

        # Remap qubit_1
        unique_q1 = torch.unique(x_n[:, 2])
        q1_map = {old_val.item(): new_val for new_val, old_val in enumerate(unique_q1)}
        for i in range(x_n.shape[0]):
            x_n[i, 2] = q1_map[x_n[i, 2].item()]

    y = [float(fidelity), float(efficiency)]

    return src, dst, x_n, y


def generate_optimization_dataset(num_samples, method='transpiler'):
    """
    Generate dataset of (unoptimized, optimized) pairs.

    Args:
        num_samples: Number of pairs to generate
        method: 'transpiler' or 'manual'
    """
    unopt_data = {'src_list': [], 'dst_list': [], 'x_n_list': [], 'y_list': []}
    opt_data = {'src_list': [], 'dst_list': [], 'x_n_list': [], 'y_list': []}

    create_pair = (create_optimization_pair_using_transpiler if method == 'transpiler'
                   else create_optimization_pair_manual)

    attempts = 0
    max_attempts = num_samples * 3  # Try up to 3x since some pairs are rejected

    pbar = tqdm(total=num_samples, desc=f"Generating pairs ({method})")

    while len(unopt_data['src_list']) < num_samples and attempts < max_attempts:
        attempts += 1

        # Random circuit parameters
        num_qubits = np.random.randint(3, 7)
        depth = np.random.randint(5, 15)
        seed = attempts

        try:
            result = create_pair(num_qubits, depth, seed)

            if result is None:
                continue

            unopt, opt, fidelity, unopt_eff, opt_eff = result

            # Convert unoptimized
            src, dst, x_n, y = circuit_to_layerdag_format(unopt, fidelity, unopt_eff)
            unopt_data['src_list'].append(src)
            unopt_data['dst_list'].append(dst)
            unopt_data['x_n_list'].append(x_n)
            unopt_data['y_list'].append(y)

            # Convert optimized
            src, dst, x_n, y = circuit_to_layerdag_format(opt, fidelity, opt_eff)
            opt_data['src_list'].append(src)
            opt_data['dst_list'].append(dst)
            opt_data['x_n_list'].append(x_n)
            opt_data['y_list'].append(y)

            pbar.update(1)

        except Exception as e:
            continue

    pbar.close()

    print(f"  Generated {len(unopt_data['src_list'])} valid pairs from {attempts} attempts")
    return unopt_data, opt_data


def main():
    np.random.seed(42)

    NUM_TRAIN = 500
    NUM_VAL = 100
    NUM_TEST = 100

    print("="*80)
    print("CREATING CIRCUIT OPTIMIZATION DATASET")
    print("="*80)
    print("\nDataset structure:")
    print("  Input: Unoptimized circuits (large, redundant)")
    print("  Target: Optimized circuits (small, efficient)")
    print("  Labels: [fidelity=0.95-1.0, efficiency_input]")
    print()

    output_dir = 'data_files/quantum_circuits_optimization'
    os.makedirs(output_dir, exist_ok=True)

    for split_name, num_samples in [('train', NUM_TRAIN), ('val', NUM_VAL), ('test', NUM_TEST)]:
        print(f"\n{split_name.upper()} SET:")

        unopt_data, opt_data = generate_optimization_dataset(num_samples, method='transpiler')

        # Save unoptimized (input)
        torch.save(unopt_data, f'{output_dir}/{split_name}_input.pth')
        print(f"  ✓ Saved input: {output_dir}/{split_name}_input.pth")

        # Save optimized (target)
        torch.save(opt_data, f'{output_dir}/{split_name}_target.pth')
        print(f"  ✓ Saved target: {output_dir}/{split_name}_target.pth")

        # Print statistics
        unopt_sizes = [len(x) for x in unopt_data['x_n_list']]
        opt_sizes = [len(x) for x in opt_data['x_n_list']]
        reduction = [(u - o) / u * 100 for u, o in zip(unopt_sizes, opt_sizes)]

        print(f"  Statistics:")
        print(f"    Unoptimized: {np.mean(unopt_sizes):.1f} ± {np.std(unopt_sizes):.1f} gates")
        print(f"    Optimized: {np.mean(opt_sizes):.1f} ± {np.std(opt_sizes):.1f} gates")
        print(f"    Avg reduction: {np.mean(reduction):.1f}%")

    print("\n" + "="*80)
    print("✓ DATASET CREATION COMPLETE")
    print("="*80)
    print(f"\nDataset saved to: {output_dir}/")
    print("\nNext step: Modify training to use input→target pairs")


if __name__ == '__main__':
    main()
