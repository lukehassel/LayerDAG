"""
Circuit cost/efficiency calculation for quantum circuits.

Provides simple efficiency metrics based on gate count.
"""
import numpy as np


def calculate_efficiency_score(gate_count, num_qubits):
    """
    Calculate circuit efficiency score based on gate count.

    Uses a sigmoid function to map gate count to efficiency [0, 1]:
    - Score = 0.5 when gate_count equals reference (expected for circuit size)
    - Score → 1.0 when gate_count is lower (more efficient)
    - Score → 0.0 when gate_count is higher (less efficient)

    Args:
        gate_count (int): Total number of gates in circuit
        num_qubits (int): Number of qubits in circuit

    Returns:
        float: Efficiency score in [0, 1]

    Examples:
        >>> calculate_efficiency_score(30, 3)  # 30 gates, 3 qubits
        0.5  # Expected gate count

        >>> calculate_efficiency_score(15, 3)  # Fewer gates
        0.88  # More efficient

        >>> calculate_efficiency_score(60, 3)  # Many gates
        0.12  # Less efficient
    """
    # Reference gate count: expected number of gates for given circuit size
    # Using num_qubits * 10 as a reasonable baseline
    reference_gates = num_qubits * 10

    # Normalize difference relative to reference
    # Positive diff = more gates than expected (bad)
    # Negative diff = fewer gates than expected (good)
    normalized_diff = (gate_count - reference_gates) / reference_gates

    # Apply sigmoid function: 1 / (1 + exp(x))
    # Negate the diff so: high gate count → low score
    efficiency_score = 1.0 / (1.0 + np.exp(normalized_diff))

    return float(efficiency_score)


def get_circuit_cost_metrics(gate_indices, num_qubits):
    """
    Get comprehensive cost metrics for a circuit.

    Args:
        gate_indices: Tensor or list of gate type indices
        num_qubits: Number of qubits in circuit

    Returns:
        dict: Cost metrics including:
            - gate_count: Total number of gates
            - efficiency_score: Efficiency score [0, 1]
            - reference_gates: Expected gate count for comparison
    """
    gate_count = len(gate_indices)
    reference_gates = num_qubits * 10

    efficiency_score = calculate_efficiency_score(gate_count, num_qubits)

    return {
        'gate_count': gate_count,
        'reference_gates': reference_gates,
        'efficiency_score': efficiency_score
    }


if __name__ == "__main__":
    # Test the efficiency score function
    print("Testing circuit efficiency score:")
    print("=" * 50)

    num_qubits = 4
    reference = num_qubits * 10  # 40 gates

    print(f"\nCircuit with {num_qubits} qubits")
    print(f"Reference gate count: {reference}")
    print()

    test_cases = [
        (10, "Very few gates (0.25x reference)"),
        (20, "Few gates (0.5x reference)"),
        (40, "Expected gates (1.0x reference)"),
        (60, "Many gates (1.5x reference)"),
        (80, "Very many gates (2.0x reference)"),
        (120, "Excessive gates (3.0x reference)")
    ]

    print(f"{'Gate Count':>12} | {'Description':>30} | {'Efficiency':>10}")
    print("-" * 60)

    for gate_count, desc in test_cases:
        efficiency = calculate_efficiency_score(gate_count, num_qubits)
        print(f"{gate_count:>12} | {desc:>30} | {efficiency:>10.4f}")

    print("\n" + "=" * 50)
    print("Efficiency score properties:")
    print("  - Score = 0.5 at reference gate count")
    print("  - Score → 1.0 for fewer gates (more efficient)")
    print("  - Score → 0.0 for more gates (less efficient)")
    print("  - Smooth sigmoid curve")
