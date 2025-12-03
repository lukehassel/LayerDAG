from qiskit import QuantumCircuit
import dgl
import torch
from mpo.circuit_utils import get_universal_gate_set

GATE_TO_IDX = {g: i for i, g in enumerate(get_universal_gate_set()['all'])}



def qasm_to_dgl(qasm_str):
    """
    Parses QASM string -> Qiskit -> DGL Graph
    """
    try:
        # 1. Parse QASM
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        
        # 2. Extract Gates and Dependencies
        # A simple approach: Create a node for every instruction
        # Edges connect sequential gates on the same qubit
        
        src_nodes = []
        dst_nodes = []
        node_gate_types = []
        node_qubit_indices = []
        
        # Track the last node ID seen on each wire
        # qc.qubits gives the list of qubit objects
        qubit_map = {q: i for i, q in enumerate(qc.qubits)}
        last_node_on_wire = {i: -1 for i in range(len(qc.qubits))}
        
        current_node_id = 0
        
        for instruction in qc.data:
            op = instruction.operation
            name = op.name
            
            # Skip barriers or measures if you only care about unitary
            if name in ['barrier', 'measure']:
                continue
                
            # Get Gate ID (default to 0 if unknown)
            g_id = GATE_TO_IDX.get(name, 0)
            
            # Identify which qubits this gate acts on
            q_indices = [qubit_map[q] for q in instruction.qubits]
            
            # For the embedding, we can just use the first qubit index 
            # (or you could expand the embedding to handle multi-qubit indices)
            primary_qubit = q_indices[0] if q_indices else 0
            
            node_gate_types.append(g_id)
            node_qubit_indices.append(primary_qubit)
            
            # Create Edges: Connect from previous gate on these wires to this gate
            for q in q_indices:
                prev_node = last_node_on_wire[q]
                if prev_node != -1:
                    src_nodes.append(prev_node)
                    dst_nodes.append(current_node_id)
                # Update tracker
                last_node_on_wire[q] = current_node_id
            
            current_node_id += 1
            
        # 3. Build DGL Graph
        if current_node_id == 0: 
            # Handle empty circuit edge case
            g = dgl.graph(([], []), num_nodes=1)
            return g, torch.tensor([0]), torch.tensor([0])

        g = dgl.graph((src_nodes, dst_nodes), num_nodes=current_node_id)
        g = dgl.add_self_loop(g) # GAT requires self-loops usually
        
        return g, torch.tensor(node_gate_types), torch.tensor(node_qubit_indices)
        
    except Exception as e:
        print(f"Error parsing QASM: {e}")
        # Return a dummy graph to prevent crashing
        g = dgl.graph(([0], [0]), num_nodes=1)