from .layer_dag import *
from .general import DAGDataset
from .tpu_tile import get_tpu_tile
from .quantum_circuit import get_quantum_circuit

def load_dataset(dataset_name):
    if dataset_name == 'tpu_tile':
        return get_tpu_tile()
    elif dataset_name == 'quantum_circuit':
        return get_quantum_circuit()
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")
