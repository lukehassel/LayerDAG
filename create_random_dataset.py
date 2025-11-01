#!/usr/bin/env python3
"""
Script to create a random graph dataset with train/val/test splits
"""
import torch
import random
import os

# Configuration
dataset_name = 'random_small'
num_train = 100
num_val = 20
num_test = 20
min_nodes = 3
max_nodes = 10

# Create output directory
output_dir = f'data_files/{dataset_name}_processed'
os.makedirs(output_dir, exist_ok=True)

def create_random_dataset(num_samples):
    """Create a random graph dataset"""
    dataset = {
        'src_list': [],
        'dst_list': [],
        'x_n_list': [],
        'y_list': []
    }

    for i in range(num_samples):
        # Random number of nodes for this graph
        num_nodes = random.randint(min_nodes, max_nodes)

        # Random number of edges (similar to TPU tile: edges ≈ nodes)
        num_edges = random.randint(num_nodes - 1, num_nodes + 3)

        # Create random DAG edges (no self-loops, no cycles)
        src = []
        dst = []
        for _ in range(num_edges):
            # Ensure DAG structure: src < dst (edges go forward)
            s = random.randint(0, num_nodes - 2)
            d = random.randint(s + 1, num_nodes - 1)
            src.append(s)
            dst.append(d)

        dataset['src_list'].append(torch.tensor(src, dtype=torch.long))
        dataset['dst_list'].append(torch.tensor(dst, dtype=torch.long))

        # Random node features (categorical: integers 0-49, matching TPU tile range)
        x_n = torch.randint(0, 50, (num_nodes,))
        dataset['x_n_list'].append(x_n)

        # Random target value
        y = random.uniform(0.0, 10.0)
        dataset['y_list'].append(y)

    return dataset

# Create train, val, test splits
print(f"Creating {dataset_name} dataset...")
print(f"  Train: {num_train} samples")
print(f"  Val: {num_val} samples")
print(f"  Test: {num_test} samples")

train_set = create_random_dataset(num_train)
val_set = create_random_dataset(num_val)
test_set = create_random_dataset(num_test)

# Save to files
train_path = os.path.join(output_dir, 'train.pth')
val_path = os.path.join(output_dir, 'val.pth')
test_path = os.path.join(output_dir, 'test.pth')

torch.save(train_set, train_path)
torch.save(val_set, val_path)
torch.save(test_set, test_path)

print(f"\nDataset saved to {output_dir}/")
print(f"  train.pth: {len(train_set['y_list'])} samples")
print(f"  val.pth: {len(val_set['y_list'])} samples")
print(f"  test.pth: {len(test_set['y_list'])} samples")
print(f"  Node range: {min_nodes}-{max_nodes} nodes per graph")
