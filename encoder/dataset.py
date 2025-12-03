import os
import torch
import torch.nn as nn
import dgl
import numpy as np
import random
import pyzx as zx
from torch.utils.data import Dataset
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt

# Qiskit Imports
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.circuit import Gate
from qiskit.converters import circuit_to_dag

# MQT Fidelity Imports
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.equivalence_checker import MPO, iterate

from src.model.diffusion import DiscreteDiffusion
from mpo.circuit_utils import get_universal_gate_set
from mpo.circuit_utils import create_random_circuit_with_universal_gates
from mpo.fidelity import get_fidelity

# Map gate names to integers for Embedding/Diffusion s
GATE_TO_IDX = {g: i for i, g in enumerate(get_universal_gate_set()['all'])}
IDX_TO_GATE = {i: g for g, i in GATE_TO_IDX.items()}
NUM_GATE_TYPES = len(get_universal_gate_set()['all'])


class EncoderDataset(Dataset):
    def __init__(self, file_path=None, size=100, min_qubits=3, max_qubits=5, 
                 min_depth=5, max_depth=15, verbose=False):
        """
        Args:
            file_path (str): Path to save/load the dataset (e.g. 'data/train_v1.pt')
            size (int): Number of pairs to generate if file doesn't exist.
        """
        self.size = size
        self.verbose = verbose
        self.file_path = file_path
        self.data_pairs = [] # List of tuples: ( (g1,t1,l1), (g2,t2,l2) )
        
        # Initialize Diffusion only if needed for generation
        self.diffusion = None
        
        # --- LOAD OR GENERATE ---
        # if self.file_path and os.path.exists(self.file_path):
        #     self._load_dataset()
        # else:
        self._init_diffusion()
        self._generate_dataset(min_qubits, max_qubits, min_depth, max_depth)
        if self.file_path:
            self._save_dataset()

    def _init_diffusion(self):
        # Uniform marginals for gate types
        marginal = torch.ones(NUM_GATE_TYPES) / NUM_GATE_TYPES
        self.diffusion = DiscreteDiffusion(marginal_list=[marginal], T=100)

    def _load_dataset(self):
        if self.verbose:
            print(f"Loading dataset from {self.file_path}...")
        self.data_pairs = torch.load(self.file_path)
        if self.verbose:
            print(f"Successfully loaded {len(self.data_pairs)} pairs.")

    def _save_dataset(self):
        if self.verbose:
            print(f"Saving dataset to {self.file_path}...")
        
        # Ensure directory exists
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.data_pairs, self.file_path)
        if self.verbose:
            print("Dataset saved.")
            
    def _apply_noise_to_circuit(self, qc, t_val=None, debug=False):
        # 1. Setup
        dag = circuit_to_dag(qc)
        ops = list(dag.topological_op_nodes())
        if not ops: return qc
        
        qubit_map = {q: i for i, q in enumerate(qc.qubits)}
        gate_indices = [GATE_TO_IDX.get(op.name, 0) for op in ops]
        x_0 = torch.tensor(gate_indices).long().unsqueeze(1)
        

        t_val = random.randint(10, 30)
        t = torch.tensor([t_val])
            
        # Noise Intensity (0.0 to 1.0)
        intensity = t.item() / self.diffusion.T
        
        # --- TUNING PARAMETERS ---
        # 1. MAX_DRIFT: How much the angle can change at t=100 (Maximum noise).
        #    0.1 rad is approx 5.7 degrees. This is enough to lower fidelity 
        #    without destroying the logic instantly.
        MAX_DRIFT = 0.1 
        
        # 2. DRIFT_PROB: Probability that a specific gate drifts.
        #    Scales with intensity. At t=10, only 10% of gates will drift.
        drift_probability = intensity 
        
        # 3. Apply Discrete Diffusion (Gate Swaps)
        _, x_t = self.diffusion.apply_noise(x_0, t)
        noisy_indices = x_t.reshape(-1).tolist()
        
        noisy_qc = QuantumCircuit(qc.num_qubits)
        
        # Sets
        univ_sets = get_universal_gate_set()
        singles = set(univ_sets['single_qubit'])
        param_gates = set(univ_sets['parametric'])

        changes = 0
        
        for i, op in enumerate(ops):
            orig_name = op.name
            new_idx = noisy_indices[i]
            new_name = IDX_TO_GATE[new_idx]
            
            # --- SELECTION LOGIC ---
            if (orig_name in singles) != (new_name in singles):
                final_name = orig_name 
            else:
                final_name = new_name 

            if final_name != orig_name: changes += 1
            
            # --- PARAMETER LOGIC ---
            current_params = []
            
            if final_name in param_gates:
                # Assuming 1 param for universal set (u3/u2 logic removed for simplicity)
                num_required = 1 
                
                # Case A: Parametric -> Parametric (Drift)
                if orig_name in param_gates and hasattr(op.op, 'params') and len(op.op.params) > 0:
                    old_vals = op.op.params
                    new_vals = []
                    
                    # PROBABILITY CHECK: Should this specific gate drift?
                    if random.random() < drift_probability:
                        # Yes, apply drift
                        sigma = MAX_DRIFT * intensity
                        for k in range(num_required):
                            base = float(old_vals[k]) if k < len(old_vals) else 0.0
                            perturbation = np.random.normal(0, sigma)
                            val = (base + perturbation) % (2 * np.pi)
                            new_vals.append(val)
                    else:
                        # No, keep exact original angle (Identity operation on params)
                        new_vals = old_vals
                        
                    current_params = new_vals
                    
                # Case B: Non-Parametric -> Parametric (Fresh Random)
                else:
                    # If a gate SWAPS to parametric, it effectively randomizes the state.
                    # We might want to dampen this too, but for now, random [0, 2pi] is correct for a "new" gate.
                    current_params = np.random.uniform(0, 2*np.pi, num_required).tolist()

            # --- APPLY ---
            qubits = [qubit_map[q] for q in op.qargs]

            try:
                # Optimization: if nothing changed, copy original object
                if final_name == orig_name and (not current_params or current_params == op.op.params):
                    noisy_qc.append(op.op, op.qargs, op.cargs)
                else:
                    getattr(noisy_qc, final_name)(*current_params, *qubits)
            except Exception as e:
                if debug: print(f"Fallback on {final_name}: {e}")
                noisy_qc.append(op.op, op.qargs, op.cargs)

        if debug:
            print(f"Noise Step t={t.item()} | Swaps: {changes}/{len(ops)}")

        return noisy_qc

    def _generate_dataset(self, min_q, max_q, min_d, max_d):
        if self.verbose:
            print(f"Generating {self.size} pairs (ZX + Noise)...")
            
        pbar = tqdm(total=self.size, disable=not self.verbose)
        attempts = 0
        
        while len(self.data_pairs) < self.size:
            attempts += 1
            if attempts > self.size * 20:
                print("\nTimeout: Could not generate enough valid pairs.")
                break
            
            # 1. Generate Random Base
            num_qubits = random.randint(min_q, max_q)
            depth = random.randint(min_d, max_d)
            circuit = create_random_circuit_with_universal_gates(num_qubits, depth)
            
            # 2. Generate ZX Variant
            # qc_zx = self._apply_zx_rules(qc_orig)
            # if qc_zx is None: continue
            
            noisy_circuit = self._apply_noise_to_circuit(circuit)
            
            #print("Original Gate Count: ", len(circuit.data))
            #print("Noisy Gate Count: ", len(noisy_circuit.data))
            
            fid_res = get_fidelity(circuit, noisy_circuit)
            
            #print("Fidelity: ", fid_res)
                
            # Extract just the numeric fidelity value from the dict
            fidelity_value = fid_res['fidelity']
            self.data_pairs.append({"circuit_1": qasm2.dumps(circuit), "circuit_2": qasm2.dumps(noisy_circuit), "fidelity": fidelity_value})
            pbar.update(1)
                
        pbar.close()

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]

# ==========================================
# Collate Function
# ==========================================
def collate_dict_batch(batch):
    """
    Custom collate function that returns dict directly for batch_size=1,
    or properly batches for larger batch sizes.
    """
    if len(batch) == 1:
        # Return the dict directly without batching
        return batch[0]
    else:
        # For larger batches, use default collation
        from torch.utils.data._utils.collate import default_collate
        return default_collate(batch)

# ==========================================
# 5. Main Execution (Test)
# ==========================================
if __name__ == "__main__":
    # Example Usage:
    # 1. Define paths
    DATA_PATH = "data/dataset.pt"
    
    # 2. Instantiate (will generate if file missing, load if present)
    dataset = EncoderDataset(file_path=DATA_PATH, size=9999999, verbose=True)
    
    # 3. Verify
    print(f"\nDataset Ready. Total Size: {len(dataset)}")
    
    # output_dir = "test_samples"
    # os.makedirs(output_dir, exist_ok=True)
     
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_dict_batch)
    
    # for i, batch in enumerate(loader):
    #     # Now we can access the dict directly without [0] indexing
    #     c1_qasm = batch["circuit_1"]
    #     c2_qasm = batch["circuit_2"]
    #     # Handle both cases: if fidelity is a dict (old format) or a number (new format)
    #     fid_val = batch["fidelity"]
    #     if isinstance(fid_val, dict):
    #         fid_val = fid_val['fidelity']
    #     elif torch.is_tensor(fid_val):
    #         fid_val = fid_val.item()

    #     print("--- Batch Info ---")
    #     # print(f"Circuit 1 (QASM len): {c1}") # [0] because batch_size=1 adds a dimension
    #     # print(f"Circuit 2 (QASM len): {c2}")
    #     print(f"Fidelity: {fid_val}")
        
    #     qc_orig = QuantumCircuit.from_qasm_str(c1_qasm)
    #     qc_noisy = QuantumCircuit.from_qasm_str(c2_qasm)
        
    #     fig, (ax1, ax2) = plt.subplots(2, 1)
        
    #     # 2. Draw the circuits onto the specific axes
    #     # We use the 'iqp' style for a clean look. Passing 'ax=' tells Qiskit where to draw.
    #     qc_orig.draw(output='mpl', style='iqp', ax=ax1)
    #     qc_noisy.draw(output='mpl', style='iqp', ax=ax2)
        
    #     # 3. Set titles for the subplots
    #     ax1.set_title(f"Original Circuit {i}", fontsize=14)
    #     ax2.set_title(f"Noisy Circuit {i} (Fidelity: {fid_val:.4f})", fontsize=14)
        
    #     # 4. Adjust layout to prevent overlaps
    #     plt.tight_layout()
        
    #     # 5. Save the combined figure
    #     output_filename = os.path.join(output_dir, f"pair_{i}_combined.png")
    #     # bbox_inches='tight' ensures labels aren't cut off
    #     plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        
    #     # 6. Close the figure to free memory
    #     plt.close(fig)
    #     print(f"Saved combined image: {output_filename}")
        
        
    #     # Stop after 1 batch for testing
    #     break