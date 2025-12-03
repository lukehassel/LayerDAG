import os
import gc
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
                 min_depth=5, max_depth=15, verbose=False,
                 chunk_size=1000):
        """
        Args:
            file_path (str): Path to save/load the dataset (e.g. 'data/train_v1.pt')
            size (int): Number of pairs to generate if file doesn't exist.
        """
        self.size = size
        self.verbose = verbose
        self.file_path = file_path

        # If file_path is a directory (or has no .pt suffix), stream data
        # to multiple chunk files to keep memory usage low.
        self.stream_to_dir = (
            self.file_path is not None
            and not self.file_path.endswith(".pt")
        )
        self.chunk_size = chunk_size
        self._current_chunk = []
        self._chunk_idx = 0

        # In in‑memory mode we keep all pairs in RAM (original behavior).
        self.data_pairs = []  # List of dicts

        # For streamed mode, track chunk metadata so __len__/__getitem__ work.
        self._chunks = []  # list of (path, length)
        self._total_len = 0
        
        # Initialize Diffusion only if needed for generation
        self.diffusion = None
        
        # --- LOAD OR GENERATE ---
        # if self.file_path and os.path.exists(self.file_path):
        #     self._load_dataset()
        # else:
        self._init_diffusion()
        self._generate_dataset(min_qubits, max_qubits, min_depth, max_depth)

        # In streaming mode, data is already flushed to disk in chunks.
        if self.file_path and not self.stream_to_dir:
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
        """Save the entire in‑memory dataset to a single .pt file."""
        if self.verbose:
            print(f"Saving dataset to {self.file_path}...")

        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.data_pairs, self.file_path)
        if self.verbose:
            print("Dataset saved.")

    def _flush_chunk_to_disk(self):
        """Write the current in‑memory chunk to disk and clear it."""
        if not self._current_chunk:
            return

        assert self.stream_to_dir, "Chunk flushing only valid in streaming mode."

        os.makedirs(self.file_path, exist_ok=True)
        chunk_path = os.path.join(
            self.file_path,
            f"dataset_chunk_{self._chunk_idx:06d}.pt",
        )
        if self.verbose:
            print(f"Flushing {len(self._current_chunk)} samples to {chunk_path}...")

        torch.save(self._current_chunk, chunk_path)

        num = len(self._current_chunk)
        self._chunks.append((chunk_path, num))
        self._total_len += num

        # Explicitly release references and ask GC to reclaim Python objects.
        self._current_chunk.clear()
        gc.collect()
        self._chunk_idx += 1
            
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

        num_generated = 0

        while num_generated < self.size:
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

            # Extract just the numeric fidelity value from the dict
            fidelity_value = fid_res['fidelity']
            sample = {
                "circuit_1": qasm2.dumps(circuit),
                "circuit_2": qasm2.dumps(noisy_circuit),
                "fidelity": fidelity_value,
            }

            if self.stream_to_dir:
                # Keep memory bounded: only hold up to chunk_size samples at once.
                self._current_chunk.append(sample)
                if len(self._current_chunk) >= self.chunk_size:
                    self._flush_chunk_to_disk()
            else:
                self.data_pairs.append(sample)

            # Drop large temporaries before next iteration.
            del circuit, noisy_circuit, fid_res, sample
            gc.collect()

            num_generated += 1
            pbar.update(1)

        # Flush any remaining samples in the last partial chunk.
        if self.stream_to_dir:
            self._flush_chunk_to_disk()

        pbar.close()

    def __len__(self):
        if self.stream_to_dir:
            return self._total_len
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if not self.stream_to_dir:
            return self.data_pairs[idx]

        # Map global index to (chunk_path, local_index) and load lazily.
        if idx < 0:
            idx = self._total_len + idx
        if idx < 0 or idx >= self._total_len:
            raise IndexError(idx)

        offset = idx
        for chunk_path, length in self._chunks:
            if offset < length:
                chunk = torch.load(chunk_path)
                return chunk[offset]
            offset -= length

        # Should not reach here
        raise IndexError(idx)

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
    DATA_DIR = "/Volumes/Samsung_T5/layerdag_dataset"  # directory on your T5

    dataset = EncoderDataset(
        file_path=DATA_DIR,     # note: directory, not a .pt file
        size=9999999,            # pick a reasonable size
        verbose=True,
        chunk_size=1000,        # how many samples per file before flushing
    )
    
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