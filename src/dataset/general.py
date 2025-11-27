import torch

from torch.utils.data import Dataset

class DAGDataset(Dataset):
    """
    Parameters
    ----------
    label : bool
        Whether each DAG has a label like runtime/latency.
    multi_dim_features : bool
        Whether node features are multi-dimensional (e.g., [gate_type, qubit_0, qubit_1]).
        If True, dummy_category will be [num_categories, -1, -1] instead of scalar.
    """
    def __init__(self, num_categories, label=False, multi_dim_features=False):
        self.src = []
        self.dst = []
        self.x_n = []

        self.label = label
        if self.label:
            self.y = []

        # For multi-dimensional features, dummy_category is a vector
        if multi_dim_features:
            # num_categories is a tensor [num_gate_types, num_qubits+2, num_qubits+2]
            # dummy_category uses the category count for each dimension (one past max index)
            # We need to increase num_categories by 1 for embeddings to accommodate the dummy category
            if isinstance(num_categories, torch.Tensor):
                self.dummy_category = [num_categories[0].item(),
                                      num_categories[1].item(),
                                      num_categories[2].item()]
                self.num_categories = num_categories + 1
            else:
                self.dummy_category = [num_categories, num_categories, num_categories]
                self.num_categories = num_categories + 1
        else:
            self.dummy_category = num_categories
            if isinstance(self.dummy_category, torch.Tensor):
                self.dummy_category = self.dummy_category.tolist()
            self.num_categories = num_categories + 1

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        if self.label:
            return self.src[index], self.dst[index], self.x_n[index], self.y[index]
        else:
            return self.src[index], self.dst[index], self.x_n[index]

    def add_data(self, src, dst, x_n, y=None):
        self.src.append(src)
        self.dst.append(dst)
        self.x_n.append(x_n)
        if (y is not None) and (self.label):
            self.y.append(y)
