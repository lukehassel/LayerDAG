"""
Test conditional generation with different efficiency labels.
This will help determine if the model can generate smaller circuits when asked.
"""
import os
import torch
import sys
sys.path.append('src')

from pprint import pprint
from setup_utils import set_seed
from src.dataset import load_dataset, DAGDataset
from src.model import DiscreteDiffusion, EdgeDiscreteDiffusion, LayerDAG

def test_conditional_sampling(model_path, device, num_samples=10):
    """
    Sample circuits with different efficiency values to test conditioning.
    """
    # Load checkpoint
    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device)

    # Create diffusion models
    node_diffusion = DiscreteDiffusion(**ckpt['node_diffusion_config']).to(device)
    edge_diffusion = EdgeDiscreteDiffusion(**ckpt['edge_diffusion_config']).to(device)

    # Create model
    model = LayerDAG(
        device=device,
        node_diffusion=node_diffusion,
        edge_diffusion=edge_diffusion,
        **ckpt['model_config']
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    print("✓ Model loaded\n")

    # Load dataset to get num_categories
    train_set, val_set, test_set = load_dataset('quantum_circuits')

    # Test different efficiency levels
    test_cases = [
        (0.3, 0.9, "Low fidelity, High efficiency (should be SMALL)"),
        (0.5, 0.9, "Medium fidelity, High efficiency (should be SMALL)"),
        (0.9, 0.9, "High fidelity, High efficiency (should be SMALL)"),
        (0.9, 0.5, "High fidelity, Medium efficiency"),
        (0.9, 0.3, "High fidelity, Low efficiency (should be LARGE)"),
    ]

    print("=" * 80)
    print("TESTING CONDITIONAL GENERATION")
    print("=" * 80)
    print(f"{'Fidelity':>10} | {'Efficiency':>10} | {'Avg Gates':>10} | {'Range':>15} | Description")
    print("-" * 80)

    for fidelity, efficiency, description in test_cases:
        # Create batch of identical labels
        raw_y_batch = [[fidelity, efficiency]] * num_samples

        # Sample
        with torch.no_grad():
            batch_edge_index, batch_x_n, batch_y = model.sample(
                device, num_samples, raw_y_batch
            )

        # Count gates
        gate_counts = [len(x_n) for x_n in batch_x_n]
        avg_gates = sum(gate_counts) / len(gate_counts)
        min_gates = min(gate_counts)
        max_gates = max(gate_counts)

        print(f"{fidelity:>10.2f} | {efficiency:>10.2f} | {avg_gates:>10.1f} | "
              f"[{min_gates:>3}, {max_gates:>3}] | {description}")

    print("-" * 80)
    print("\nInterpretation:")
    print("  - If avg gates decreases with higher efficiency → Model learns conditioning ✓")
    print("  - If avg gates stays same → Conditioning is weak ✗")
    print("=" * 80)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    test_conditional_sampling(args.model_path, device, args.num_samples)
