import os
import torch
import numpy as np
import sys
sys.path.append('src')

from collections import defaultdict
from src.dataset import load_dataset
import matplotlib
matplotlib.use('Agg')  # For saving plots without display
import matplotlib.pyplot as plt

def load_generated_circuits(file_path):
    """Load generated circuits from .pth file."""
    print(f"Loading generated circuits from: {file_path}")
    data = torch.load(file_path)

    circuits = []
    for i in range(len(data['src_list'])):
        circuits.append({
            'src': data['src_list'][i],
            'dst': data['dst_list'][i],
            'x_n': data['x_n_list'][i],
            'y': data['y_list'][i]
        })

    print(f"  Loaded {len(circuits)} circuits")
    return circuits

def compute_validity_metrics(circuits):
    """
    Compute validity metrics for generated circuits.

    Checks:
    1. Valid DAG structure (no cycles)
    2. Valid gate types (0-17)
    3. Valid qubit indices (consecutive from 0)
    """
    print("\n" + "="*60)
    print("VALIDITY METRICS")
    print("="*60)

    num_circuits = len(circuits)
    num_valid_dags = 0
    num_valid_gates = 0
    num_valid_qubits = 0
    num_fully_valid = 0

    invalid_examples = []

    for i, circuit in enumerate(circuits):
        src = circuit['src']
        dst = circuit['dst']
        x_n = circuit['x_n']

        is_valid_dag = True
        is_valid_gates = True
        is_valid_qubits = True

        # Check 1: Valid DAG (dst > src for all edges)
        if len(src) > 0:
            if not torch.all(dst > src):
                is_valid_dag = False
                if len(invalid_examples) < 5:
                    invalid_examples.append(f"Circuit {i}: Invalid DAG (has cycles)")

        # Check 2: Valid gate types (0-17)
        gate_types = x_n[:, 0]
        if torch.any(gate_types < 0) or torch.any(gate_types > 17):
            is_valid_gates = False
            if len(invalid_examples) < 5:
                invalid_examples.append(
                    f"Circuit {i}: Invalid gate types "
                    f"(range [{gate_types.min().item()}, {gate_types.max().item()}])"
                )

        # Check 3: Valid qubit indices (consecutive from 0)
        qubit_0 = x_n[:, 1]
        qubit_1 = x_n[:, 2]

        # Check qubit_0 consecutive
        unique_q0 = torch.unique(qubit_0)
        expected_q0 = torch.arange(len(unique_q0))
        if not torch.equal(torch.sort(unique_q0)[0], expected_q0):
            is_valid_qubits = False
            if len(invalid_examples) < 5:
                invalid_examples.append(f"Circuit {i}: Non-consecutive qubit_0 indices")

        # Check qubit_1 consecutive (excluding 0 for single-qubit gates)
        unique_q1 = torch.unique(qubit_1[qubit_1 > 0])  # Exclude 0s
        if len(unique_q1) > 0:
            # For qubit_1, we expect 0, 1, 2, ... but 0 is used for single-qubit gates
            # So we check the non-zero values
            max_q1 = unique_q1.max().item()
            if max_q1 >= len(x_n):
                is_valid_qubits = False
                if len(invalid_examples) < 5:
                    invalid_examples.append(f"Circuit {i}: Invalid qubit_1 indices")

        # Count valid circuits
        if is_valid_dag:
            num_valid_dags += 1
        if is_valid_gates:
            num_valid_gates += 1
        if is_valid_qubits:
            num_valid_qubits += 1
        if is_valid_dag and is_valid_gates and is_valid_qubits:
            num_fully_valid += 1

    # Print results
    print(f"\nTotal circuits: {num_circuits}")
    print(f"\nValidity Checks:")
    print(f"  ✓ Valid DAG structure:     {num_valid_dags:4d} / {num_circuits} ({100*num_valid_dags/num_circuits:.1f}%)")
    print(f"  ✓ Valid gate types (0-17): {num_valid_gates:4d} / {num_circuits} ({100*num_valid_gates/num_circuits:.1f}%)")
    print(f"  ✓ Valid qubit indices:     {num_valid_qubits:4d} / {num_circuits} ({100*num_valid_qubits/num_circuits:.1f}%)")
    print(f"\n  → Fully valid circuits:    {num_fully_valid:4d} / {num_circuits} ({100*num_fully_valid/num_circuits:.1f}%)")

    if invalid_examples:
        print(f"\nExample invalid circuits (showing first {len(invalid_examples)}):")
        for example in invalid_examples:
            print(f"  • {example}")

    return {
        'num_circuits': num_circuits,
        'num_valid_dags': num_valid_dags,
        'num_valid_gates': num_valid_gates,
        'num_valid_qubits': num_valid_qubits,
        'num_fully_valid': num_fully_valid,
        'validity_rate': num_fully_valid / num_circuits if num_circuits > 0 else 0
    }

def compute_quality_metrics(real_circuits, gen_circuits):
    """
    Compute quality metrics comparing real and generated circuits.

    Metrics:
    1. Label distribution matching (fidelity and efficiency)
    2. Circuit size distribution (number of gates)
    3. Edge density distribution
    """
    print("\n" + "="*60)
    print("QUALITY METRICS")
    print("="*60)

    # Extract labels
    real_fidelity = [c['y'][0] if isinstance(c['y'][0], float) else c['y'][0].item() for c in real_circuits]
    real_efficiency = [c['y'][1] if isinstance(c['y'][1], float) else c['y'][1].item() for c in real_circuits]
    gen_fidelity = [c['y'][0] if isinstance(c['y'][0], float) else c['y'][0].item() for c in gen_circuits]
    gen_efficiency = [c['y'][1] if isinstance(c['y'][1], float) else c['y'][1].item() for c in gen_circuits]

    # Extract circuit sizes
    real_sizes = [len(c['x_n']) for c in real_circuits]
    gen_sizes = [len(c['x_n']) for c in gen_circuits]

    # Extract edge counts
    real_edges = [len(c['src']) for c in real_circuits]
    gen_edges = [len(c['src']) for c in gen_circuits]

    # Compute statistics
    print(f"\nLabel Distribution:")
    print(f"  Fidelity:")
    print(f"    Real: mean={np.mean(real_fidelity):.3f}, std={np.std(real_fidelity):.3f}")
    print(f"    Gen:  mean={np.mean(gen_fidelity):.3f}, std={np.std(gen_fidelity):.3f}")
    print(f"    MAE:  {np.mean(np.abs(np.array(real_fidelity) - np.array(gen_fidelity))):.3f}")

    print(f"\n  Efficiency:")
    print(f"    Real: mean={np.mean(real_efficiency):.3f}, std={np.std(real_efficiency):.3f}")
    print(f"    Gen:  mean={np.mean(gen_efficiency):.3f}, std={np.std(gen_efficiency):.3f}")
    print(f"    MAE:  {np.mean(np.abs(np.array(real_efficiency) - np.array(gen_efficiency))):.3f}")

    print(f"\nCircuit Size Distribution:")
    print(f"  Real: mean={np.mean(real_sizes):.1f}, std={np.std(real_sizes):.1f}, range=[{min(real_sizes)}, {max(real_sizes)}]")
    print(f"  Gen:  mean={np.mean(gen_sizes):.1f}, std={np.std(gen_sizes):.1f}, range=[{min(gen_sizes)}, {max(gen_sizes)}]")

    print(f"\nEdge Count Distribution:")
    print(f"  Real: mean={np.mean(real_edges):.1f}, std={np.std(real_edges):.1f}, range=[{min(real_edges)}, {max(real_edges)}]")
    print(f"  Gen:  mean={np.mean(gen_edges):.1f}, std={np.std(gen_edges):.1f}, range=[{min(gen_edges)}, {max(gen_edges)}]")

    return {
        'real_fidelity': real_fidelity,
        'gen_fidelity': gen_fidelity,
        'real_efficiency': real_efficiency,
        'gen_efficiency': gen_efficiency,
        'real_sizes': real_sizes,
        'gen_sizes': gen_sizes,
        'real_edges': real_edges,
        'gen_edges': gen_edges
    }

def compute_diversity_metrics(circuits):
    """
    Compute diversity metrics for generated circuits.

    Metrics:
    1. Uniqueness rate
    2. Gate type distribution
    """
    print("\n" + "="*60)
    print("DIVERSITY METRICS")
    print("="*60)

    # Compute circuit hashes for uniqueness
    circuit_hashes = set()
    for circuit in circuits:
        # Create a hash from src, dst, and x_n
        circuit_str = (
            f"{circuit['src'].tolist()}-"
            f"{circuit['dst'].tolist()}-"
            f"{circuit['x_n'].tolist()}"
        )
        circuit_hashes.add(circuit_str)

    uniqueness = len(circuit_hashes) / len(circuits) if len(circuits) > 0 else 0

    # Compute gate type distribution
    gate_counts = defaultdict(int)
    total_gates = 0

    for circuit in circuits:
        gate_types = circuit['x_n'][:, 0]
        for gate_type in gate_types:
            gate_counts[gate_type.item()] += 1
            total_gates += 1

    # Map gate indices to names
    gate_names = ['cp', 'cx', 'cz', 'h', 'id', 'p', 'rx', 'rxx', 'ry', 'ryy',
                  'rz', 'rzz', 'swap', 'sx', 'u', 'x', 'y', 'z']

    print(f"\nUniqueness: {len(circuit_hashes)} / {len(circuits)} ({100*uniqueness:.1f}%)")

    print(f"\nGate Type Distribution (top 10):")
    sorted_gates = sorted(gate_counts.items(), key=lambda x: x[1], reverse=True)
    for gate_idx, count in sorted_gates[:10]:
        gate_name = gate_names[gate_idx] if gate_idx < len(gate_names) else f"unknown_{gate_idx}"
        percentage = 100 * count / total_gates if total_gates > 0 else 0
        print(f"  {gate_name:6s}: {count:5d} ({percentage:5.1f}%)")

    return {
        'uniqueness': uniqueness,
        'num_unique': len(circuit_hashes),
        'gate_counts': dict(gate_counts)
    }

def create_visualizations(quality_metrics, output_dir):
    """Create visualization plots comparing real and generated circuits."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Quantum Circuits: Real vs Generated', fontsize=16)

    # Plot 1: Fidelity distribution
    axes[0, 0].hist(quality_metrics['real_fidelity'], bins=20, alpha=0.5,
                    label='Real', color='blue', density=True)
    axes[0, 0].hist(quality_metrics['gen_fidelity'], bins=20, alpha=0.5,
                    label='Generated', color='red', density=True)
    axes[0, 0].set_xlabel('Fidelity')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Fidelity Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Efficiency distribution
    axes[0, 1].hist(quality_metrics['real_efficiency'], bins=20, alpha=0.5,
                    label='Real', color='blue', density=True)
    axes[0, 1].hist(quality_metrics['gen_efficiency'], bins=20, alpha=0.5,
                    label='Generated', color='red', density=True)
    axes[0, 1].set_xlabel('Efficiency')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Efficiency Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Circuit size distribution
    axes[0, 2].hist(quality_metrics['real_sizes'], bins=20, alpha=0.5,
                    label='Real', color='blue', density=True)
    axes[0, 2].hist(quality_metrics['gen_sizes'], bins=20, alpha=0.5,
                    label='Generated', color='red', density=True)
    axes[0, 2].set_xlabel('Number of Gates')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Circuit Size Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Fidelity scatter (real vs gen)
    axes[1, 0].scatter(quality_metrics['real_fidelity'],
                       quality_metrics['gen_fidelity'],
                       alpha=0.5, s=20)
    axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect match')
    axes[1, 0].set_xlabel('Real Fidelity')
    axes[1, 0].set_ylabel('Generated Fidelity')
    axes[1, 0].set_title('Fidelity: Real vs Generated')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Efficiency scatter (real vs gen)
    axes[1, 1].scatter(quality_metrics['real_efficiency'],
                       quality_metrics['gen_efficiency'],
                       alpha=0.5, s=20)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Perfect match')
    axes[1, 1].set_xlabel('Real Efficiency')
    axes[1, 1].set_ylabel('Generated Efficiency')
    axes[1, 1].set_title('Efficiency: Real vs Generated')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Edge count distribution
    axes[1, 2].hist(quality_metrics['real_edges'], bins=20, alpha=0.5,
                    label='Real', color='blue', density=True)
    axes[1, 2].hist(quality_metrics['gen_edges'], bins=20, alpha=0.5,
                    label='Generated', color='red', density=True)
    axes[1, 2].set_xlabel('Number of Edges')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Edge Count Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'evaluation_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plots to: {plot_path}")
    plt.close()

def save_evaluation_report(validity_metrics, quality_metrics, diversity_metrics, output_dir):
    """Save evaluation metrics to a text file."""
    report_path = os.path.join(output_dir, 'evaluation_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("QUANTUM CIRCUITS EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")

        f.write("VALIDITY METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Total circuits: {validity_metrics['num_circuits']}\n")
        f.write(f"Valid DAGs: {validity_metrics['num_valid_dags']} ({100*validity_metrics['num_valid_dags']/validity_metrics['num_circuits']:.1f}%)\n")
        f.write(f"Valid gates: {validity_metrics['num_valid_gates']} ({100*validity_metrics['num_valid_gates']/validity_metrics['num_circuits']:.1f}%)\n")
        f.write(f"Valid qubits: {validity_metrics['num_valid_qubits']} ({100*validity_metrics['num_valid_qubits']/validity_metrics['num_circuits']:.1f}%)\n")
        f.write(f"Fully valid: {validity_metrics['num_fully_valid']} ({100*validity_metrics['validity_rate']:.1f}%)\n\n")

        f.write("QUALITY METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Fidelity MAE: {np.mean(np.abs(np.array(quality_metrics['real_fidelity']) - np.array(quality_metrics['gen_fidelity']))):.3f}\n")
        f.write(f"Efficiency MAE: {np.mean(np.abs(np.array(quality_metrics['real_efficiency']) - np.array(quality_metrics['gen_efficiency']))):.3f}\n")
        f.write(f"Size difference: {abs(np.mean(quality_metrics['real_sizes']) - np.mean(quality_metrics['gen_sizes'])):.1f} gates\n\n")

        f.write("DIVERSITY METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Uniqueness: {diversity_metrics['uniqueness']*100:.1f}%\n")
        f.write(f"Unique circuits: {diversity_metrics['num_unique']} / {validity_metrics['num_circuits']}\n")

    print(f"✓ Saved report to: {report_path}")

def main(args):
    print("\n" + "="*60)
    print("QUANTUM CIRCUITS EVALUATION")
    print("="*60)

    # Load real circuits
    print(f"\nLoading real circuits...")
    train_set, val_set, test_set = load_dataset('quantum_circuits')

    if args.dataset == 'validation':
        real_dataset = val_set
    elif args.dataset == 'test':
        real_dataset = test_set
    elif args.dataset == 'train':
        real_dataset = train_set
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    real_circuits = []
    for i in range(len(real_dataset)):
        src, dst, x_n, y = real_dataset[i]
        real_circuits.append({'src': src, 'dst': dst, 'x_n': x_n, 'y': y})

    print(f"  Loaded {len(real_circuits)} real circuits from {args.dataset} set")

    # Load generated circuits
    gen_circuits = load_generated_circuits(args.generated_file)

    # Compute metrics
    validity_metrics = compute_validity_metrics(gen_circuits)
    quality_metrics = compute_quality_metrics(real_circuits, gen_circuits)
    diversity_metrics = compute_diversity_metrics(gen_circuits)

    # Create visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    create_visualizations(quality_metrics, args.output_dir)

    # Save report
    save_evaluation_report(validity_metrics, quality_metrics, diversity_metrics, args.output_dir)

    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - evaluation_plots.png")
    print(f"  - evaluation_report.txt\n")

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Evaluate generated quantum circuits")

    parser.add_argument("--generated_file", type=str, required=True,
                        help="Path to generated circuits .pth file")
    parser.add_argument("--dataset", type=str, default="validation",
                        choices=['train', 'validation', 'test'],
                        help="Which dataset to compare against")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")

    args = parser.parse_args()

    main(args)
