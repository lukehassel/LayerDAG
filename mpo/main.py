"""
Measure the memory requirement after 100 runs of the get_fidelity function
to check for potential memory leaks.
"""
import sys
sys.path.append('..')

import tracemalloc
import time  # To show progress
from circuit_utils import create_random_circuit_with_universal_gates
from fidelity import get_fidelity
import opt_einsum


def main():
    print("=" * 70)
    print("Measuring Memory Requirement After 100 Runs of get_fidelity()")
    print("=" * 70)

    # Parameters to create test circuits
    num_qubits = 4
    circuit_depth = 8
    num_runs = 500

    print(f"\nCreating test circuits:")
    print(f"  - Number of qubits: {num_qubits}")
    print(f"  - Circuit depth: {circuit_depth}")

    # Create two different circuits to compare
    # Using different seeds to ensure they are not identical
    circuit1 = create_random_circuit_with_universal_gates(
        num_qubits=num_qubits,
        depth=circuit_depth,
        seed=42,
        max_operands=2
    )

    circuit2 = create_random_circuit_with_universal_gates(
        num_qubits=num_qubits,
        depth=circuit_depth,
        seed=101,
        max_operands=2
    )
    print("... Test circuits created.")
    print(f"Will run get_fidelity() {num_runs} times.")

    # --- Memory Measurement ---

    print("\nStarting memory trace...")
    tracemalloc.start()
    
    # Clear any existing traces before starting
    tracemalloc.clear_traces()
    
    # Take a snapshot *before* the loop
    snapshot_before_loop = tracemalloc.take_snapshot()

    print(f"Running loop ({num_runs} iterations)...")
    start_time = time.time()
    
    for i in range(num_runs):
        # We call the function, but don't need to store the result 100 times
        _ = get_fidelity(circuit1, circuit2)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"  ... completed run {i+1}/{num_runs}")

    end_time = time.time()
    print(f"... Loop finished in {end_time - start_time:.2f} seconds.")

    # Take a snapshot *after* the entire loop
    snapshot_after_loop = tracemalloc.take_snapshot()

    # Stop tracing
    tracemalloc.stop()
    print("... Memory trace stopped.")

    # --- Results ---

    print("\n" + "=" * 70)
    print("Memory Analysis Results (After 100 Runs)")
    print("=" * 70)

    # Compare the final snapshot to the one before the loop
    top_stats = snapshot_after_loop.compare_to(snapshot_before_loop, 'lineno')

    # Calculate total accumulated memory
    total_accumulated = sum(stat.size_diff for stat in top_stats)

    print(f"Total accumulated memory after {num_runs} runs: {total_accumulated / 1024:.2f} KiB")
    
    if total_accumulated > 0:
        print(f"Average memory increase per run: {(total_accumulated / num_runs) / 1024:.4f} KiB")
    
    print("\nThis value represents the total memory that was allocated and *not* freed")
    print("after 100 executions. A value close to zero is ideal.")

    # Only show details if a significant leak (e.g., > 1 KiB) is detected
    if total_accumulated > 1024:
        print("\nTop 10 new allocations (potential leaks):")
        for stat in top_stats[:10]:
            print(stat)
    else:
        print("\nNo significant memory accumulation detected.")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()