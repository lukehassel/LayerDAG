import os
import torch

from .general import DAGDataset

def to_dag_dataset(data_dict, num_categories, multi_dim_features=False):
    dataset = DAGDataset(num_categories=num_categories, label=True, multi_dim_features=multi_dim_features)

    src_list = data_dict['src_list']
    dst_list = data_dict['dst_list']
    x_n_list = data_dict['x_n_list']
    y_list = data_dict['y_list']

    num_g = len(src_list)
    for i in range(num_g):
        dataset.add_data(src_list[i],
                         dst_list[i],
                         x_n_list[i],
                         y_list[i])

    return dataset

def get_quantum_circuits():
    """
    Load quantum circuits dataset with fidelity labels.

    Dataset structure:
    - src_list: Edge source nodes (gate dependencies)
    - dst_list: Edge destination nodes
    - x_n_list: Node features (gate type indices)
    - y_list: Target labels (fidelity values 0-1)

    Returns:
        tuple: (train_set, val_set, test_set) as DAGDataset objects
    """
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(root_path, '../../data_files/quantum_circuits_processed')

    train_path = os.path.join(root_path, 'train.pth')
    val_path = os.path.join(root_path, 'val.pth')
    test_path = os.path.join(root_path, 'test.pth')

    print('Loading Quantum Circuits dataset...')
    train_set = torch.load(train_path)
    val_set = torch.load(val_path)
    test_set = torch.load(test_path)

    # Get number of categories for each feature dimension
    # Node features are [gate_type, qubit_0, qubit_1] for multi-dim, or scalar for 1D
    # Note: Preprocessing (shifting and normalization of qubit indices) is now done during
    # dataset generation in create_quantum_circuit_dataset.py, so the loaded data is
    # already in correct format with consecutive integers starting from 0.
    # IMPORTANT: Check all splits (train, val, test) to get the global maximum for each dimension
    all_x_n = torch.cat(train_set['x_n_list'] + val_set['x_n_list'] + test_set['x_n_list'])
    multi_dim = all_x_n.ndim == 2

    if multi_dim:
        # Multi-dimensional features: data is already preprocessed with consecutive integers
        # Column 0: gate types (0, 1, 2, ..., max)
        # Column 1: qubit_0 indices (0, 1, 2, ..., max)
        # Column 2: qubit_1 indices (0, 1, 2, ..., max)
        num_gate_types = all_x_n[:, 0].max().item() + 1
        num_qubit_indices_0 = all_x_n[:, 1].max().item() + 1
        num_qubit_indices_1 = all_x_n[:, 2].max().item() + 1
        num_categories = torch.tensor([num_gate_types, num_qubit_indices_0, num_qubit_indices_1])
    else:
        # Scalar features
        num_categories = all_x_n.max().item() + 1

    # Convert to DAGDataset format
    train_set = to_dag_dataset(train_set, num_categories, multi_dim_features=multi_dim)
    val_set = to_dag_dataset(val_set, num_categories, multi_dim_features=multi_dim)
    test_set = to_dag_dataset(test_set, num_categories, multi_dim_features=multi_dim)

    print(f'  Num categories (gate types): {num_categories + 1}')
    print(f'  Train samples: {len(train_set)}')
    print(f'  Val samples: {len(val_set)}')
    print(f'  Test samples: {len(test_set)}')

    return train_set, val_set, test_set
