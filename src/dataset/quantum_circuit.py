import torch
import os
from .general import DAGDataset

def get_quantum_circuit():
    """Load quantum circuit datasets from .pth files."""

    # Use direct path to dataset within workspace
    dataset_dir = os.path.expanduser('~/dataset/layerdag_processed')

    # Load data
    train_data = torch.load(os.path.join(dataset_dir, 'train.pth'))
    val_data = torch.load(os.path.join(dataset_dir, 'val.pth'))
    test_data = torch.load(os.path.join(dataset_dir, 'test.pth'))

    # Find all unique node IDs and create consecutive mapping
    all_nodes = []
    for x_n_list in [train_data['x_n_list'], val_data['x_n_list'], test_data['x_n_list']]:
        for x_n in x_n_list:
            all_nodes.extend(x_n)

    unique_nodes = sorted(set(all_nodes))
    # Create mapping from original IDs to consecutive IDs (0, 1, 2, ...)
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    num_categories = len(unique_nodes) - 1  # -1 because we don't count the dummy category

    # Create datasets
    train_set = DAGDataset(num_categories, label=True)
    val_set = DAGDataset(num_categories, label=True)
    test_set = DAGDataset(num_categories, label=True)

    # Helper function to remap node IDs
    def remap_nodes(x_n):
        return torch.tensor([node_id_map[int(node)] for node in x_n])

    # Add training data (convert lists to tensors and remap node IDs)
    for src, dst, x_n, y in zip(train_data['src_list'], train_data['dst_list'],
                                  train_data['x_n_list'], train_data['y_list']):
        train_set.add_data(torch.tensor(src), torch.tensor(dst), remap_nodes(x_n), y)

    # Add validation data (convert lists to tensors and remap node IDs)
    for src, dst, x_n, y in zip(val_data['src_list'], val_data['dst_list'],
                                  val_data['x_n_list'], val_data['y_list']):
        val_set.add_data(torch.tensor(src), torch.tensor(dst), remap_nodes(x_n), y)

    # Add test data (convert lists to tensors and remap node IDs)
    for src, dst, x_n, y in zip(test_data['src_list'], test_data['dst_list'],
                                  test_data['x_n_list'], test_data['y_list']):
        test_set.add_data(torch.tensor(src), torch.tensor(dst), remap_nodes(x_n), y)

    return train_set, val_set, test_set
