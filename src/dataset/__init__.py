from .layer_dag import *
from .general import DAGDataset
from .tpu_tile import get_tpu_tile, get_random_small

def load_dataset(dataset_name):
    if dataset_name == 'tpu_tile':
        return get_tpu_tile()
    elif dataset_name == 'random_small':
        return get_random_small()
    else:
        return NotImplementedError
