"""
Modified training script for circuit optimization.

Key difference from standard training:
- Standard: Train to denoise random noise → clean circuit
- Optimization: Train to map unoptimized → optimized circuit

The model architecture stays the same!
Only the training data and objective change.
"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm

from src.dataset import DAGDataset, load_dataset
from src.model import DiscreteDiffusion, EdgeDiscreteDiffusion, LayerDAG


def load_optimization_dataset():
    """
    Load optimization dataset with input→target pairs.

    Returns:
        (input_train, target_train), (input_val, target_val), (input_test, target_test)
    """
    base_path = 'data_files/quantum_circuits_optimization'

    # Load input circuits (unoptimized)
    train_input = torch.load(f'{base_path}/train_input.pth')
    val_input = torch.load(f'{base_path}/val_input.pth')
    test_input = torch.load(f'{base_path}/test_input.pth')

    # Load target circuits (optimized)
    train_target = torch.load(f'{base_path}/train_target.pth')
    val_target = torch.load(f'{base_path}/val_target.pth')
    test_target = torch.load(f'{base_path}/test_target.pth')

    # Get num_categories from data
    all_x_n = torch.cat(train_input['x_n_list'] + val_input['x_n_list'] + test_input['x_n_list'])
    num_categories = torch.tensor([
        all_x_n[:, 0].max().item() + 1,  # gate types
        all_x_n[:, 1].max().item() + 1,  # qubit_0
        all_x_n[:, 2].max().item() + 1   # qubit_1
    ])

    print(f'Num categories: {num_categories}')

    # Convert to DAGDataset
    def to_dataset(data_dict, num_cat):
        dataset = DAGDataset(num_cat, label=True, multi_dim_features=True)
        for i in range(len(data_dict['src_list'])):
            dataset.add_data(
                data_dict['src_list'][i],
                data_dict['dst_list'][i],
                data_dict['x_n_list'][i],
                data_dict['y_list'][i]
            )
        return dataset

    train_input_set = to_dataset(train_input, num_categories)
    train_target_set = to_dataset(train_target, num_categories)
    val_input_set = to_dataset(val_input, num_categories)
    val_target_set = to_dataset(val_target, num_categories)
    test_input_set = to_dataset(test_input, num_categories)
    test_target_set = to_dataset(test_target, num_categories)

    return (train_input_set, train_target_set), \
           (val_input_set, val_target_set), \
           (test_input_set, test_target_set), \
           num_categories


def create_paired_dataloader(input_set, target_set, batch_size, shuffle=True):
    """
    Create dataloader that yields (input_batch, target_batch) pairs.

    This is the KEY difference: we pair unoptimized inputs with optimized targets.
    """
    # We'll manually batch to ensure input-target pairing
    indices = list(range(len(input_set)))
    if shuffle:
        import random
        random.shuffle(indices)

    batches = []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i+batch_size]
        batches.append(batch_idx)

    return batches, input_set, target_set


def optimization_training_step(model, input_batch_data, target_batch_data, criterion, optimizer, device):
    """
    One training step for optimization.

    Args:
        input_batch_data: Unoptimized circuits
        target_batch_data: Optimized circuits (what we want to produce)

    Key insight:
    - Input has labels [fidelity=0.99, efficiency=0.3]  (unoptimized)
    - Target has labels [fidelity=0.99, efficiency=0.8] (optimized)
    - We want model to learn: given input, produce target
    """

    # Unpack input (unoptimized circuits)
    input_src, input_dst, input_x_n, input_y = input_batch_data

    # Unpack target (optimized circuits)
    target_src, target_dst, target_x_n, target_y = target_batch_data

    # Move to device
    # ... (similar to current training)

    # MODIFIED OBJECTIVE:
    # Instead of: model(noisy) → original
    # We do: model(unoptimized, target_labels) → optimized

    # The KEY: We condition on TARGET labels (high efficiency)
    # This tells model: "I want high efficiency output"

    # During node prediction phase:
    # Input: unoptimized_circuit
    # Conditioning: target_y (high efficiency)
    # Target: optimized_circuit

    # The model learns:
    # "When I see redundant circuit + request for high efficiency
    #  → Output the optimized version"

    loss = compute_loss(...)  # Similar to current training
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    print("="*80)
    print("TRAINING CIRCUIT OPTIMIZATION MODEL")
    print("="*80)

    # Load optimization dataset
    (train_in, train_tgt), (val_in, val_tgt), (test_in, test_tgt), num_cat = load_optimization_dataset()

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_in)} pairs")
    print(f"  Val: {len(val_in)} pairs")
    print(f"  Test: {len(test_in)} pairs")

    # Create model (same architecture!)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ... model creation same as before ...

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        model.train()
        batches, input_set, target_set = create_paired_dataloader(train_in, train_tgt, batch_size=32)

        for batch_idx in tqdm(batches):
            # Get paired data
            input_batch = [input_set[i] for i in batch_idx]
            target_batch = [target_set[i] for i in batch_idx]

            # Training step
            loss = optimization_training_step(
                model, input_batch, target_batch,
                criterion, optimizer, device
            )

        print(f"  Train loss: {loss:.4f}")

    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
