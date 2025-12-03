import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv, GlobalAttentionPooling
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# ==========================================
# 1. The Model: Fidelity Encoder
# ==========================================
class FidelityEncoder(nn.Module):
    def __init__(self, num_gate_types, max_qubits=20, hidden_dim=128, num_layers=3, num_heads=4):
        super().__init__()
        
        # Embeddings
        self.gate_embedding = nn.Embedding(num_gate_types, hidden_dim)
        self.qubit_embedding = nn.Embedding(max_qubits, hidden_dim)

        # GNN Backbone (GAT)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, num_heads=num_heads, 
                        feat_drop=0.1, attn_drop=0.1, residual=True, allow_zero_in_degree=True)
            )
            
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Attention Pooling (Readout)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        self.pooling = GlobalAttentionPooling(self.pool_gate)
        
        # Projection Head (Standard in Contrastive Learning - maps to latent space)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64) # Final embedding size
        )

    def forward(self, g, gate_types, qubit_indices):
        # 1. Input Embedding
        h_type = self.gate_embedding(gate_types)
        h_loc = self.qubit_embedding(qubit_indices)
        h = h_type + h_loc 
        
        # 2. Graph Encoding
        for layer in self.layers:
            # Flatten heads: (N, Heads, Dim) -> (N, Heads*Dim)
            h = layer(g, h).flatten(1) 
            h = F.relu(h)
            
        h = self.norm(h)
        
        # 3. Pooling (Graph Representation)
        z_graph = self.pooling(g, h)
        
        # 4. Projection to Latent Space
        z_proj = self.projection(z_graph)
        return z_proj

# ==========================================
# 2. The Loss: Supervised InfoNCE
# ==========================================
class CircuitInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: Embeddings of the first view (Original Circuits) [Batch, Dim]
        z_j: Embeddings of the second view (Equivalent Circuits) [Batch, Dim]
        """
        batch_size = z_i.shape[0]
        
        # Concatenate all features: [2*Batch, Dim]
        features = torch.cat([z_i, z_j], dim=0)
        
        # Normalize to unit sphere
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix: [2N, 2N]
        similarity_matrix = torch.matmul(features, features.T)
        
        # Labels: z_i[k] should match z_j[k]
        # Create a mask for positive pairs
        labels = torch.cat([
            torch.arange(batch_size) + batch_size, # i matches i+N
            torch.arange(batch_size)               # i+N matches i
        ]).to(features.device)
        
        # We only need the similarity of the "target" class
        # This implementation simplifies standard InfoNCE for paired batches
        
        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(similarity_matrix), 
            1, 
            torch.arange(2 * batch_size).view(-1, 1).to(features.device), 
            0
        )
        
        # Mask for positives
        mask = torch.zeros_like(similarity_matrix)
        for i in range(batch_size):
            mask[i, i + batch_size] = 1
            mask[i + batch_size, i] = 1
            
        # Compute Logits
        logits = similarity_matrix / self.temperature
        
        # For each element, we want to maximize the log-probability of the positive pair
        # relative to all other pairs.
        # Standard Cross Entropy Loss can be used here if we treat it as classification
        
        loss = F.cross_entropy(logits, labels)
        
        return loss

# ==========================================
# 3. Data Simulation (Mocking Qiskit/DAGs)
# ==========================================
def build_dgl_graph(gate_list, num_qubits):
    """
    Converts a list of gates into a DGL DAG based on wire dependencies.
    gate_list: list of tuples (gate_id, target_qubit)
    """
    src_nodes = []
    dst_nodes = []
    
    # Track the last node seen on each qubit wire
    # Initialize with -1 (or virtual start nodes if preferred)
    last_node_on_wire = {i: -1 for i in range(num_qubits)}
    
    node_gate_types = []
    node_qubit_indices = []
    
    current_node_id = 0
    
    for gate_id, qubit in gate_list:
        node_gate_types.append(gate_id)
        node_qubit_indices.append(qubit)
        
        # If there was a previous gate on this wire, add an edge
        if last_node_on_wire[qubit] != -1:
            src_nodes.append(last_node_on_wire[qubit])
            dst_nodes.append(current_node_id)
            
        # Update tracker
        last_node_on_wire[qubit] = current_node_id
        current_node_id += 1
        
    # Create DGL Graph
    g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(gate_list))
    
    # Add self-loops to ensure message passing works even for isolated nodes
    g = dgl.add_self_loop(g)
    
    return g, torch.tensor(node_gate_types), torch.tensor(node_qubit_indices)

class SyntheticQuantumDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.num_gate_types = 5 # e.g., 0:H, 1:X, 2:CNOT...
        self.num_qubits = 5
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 1. Generate Random Circuit A
        length = random.randint(5, 15)
        circuit_A_gates = []
        for _ in range(length):
            g_id = random.randint(0, self.num_gate_types - 1)
            q_id = random.randint(0, self.num_qubits - 1)
            circuit_A_gates.append((g_id, q_id))
            
        # 2. Generate Equivalent Circuit B (Mock Equivalence)
        # In reality, you would use Qiskit transpiler here.
        # For this demo, we swap two gates or insert an Identity (mocked as no-op)
        circuit_B_gates = circuit_A_gates.copy()
        if len(circuit_B_gates) > 2:
            # Trivial Commutation simulation: Swap two adjacent gates
            idx = random.randint(0, len(circuit_B_gates)-2)
            circuit_B_gates[idx], circuit_B_gates[idx+1] = circuit_B_gates[idx+1], circuit_B_gates[idx]
            
        # 3. Build Graphs
        g1, types1, locs1 = build_dgl_graph(circuit_A_gates, self.num_qubits)
        g2, types2, locs2 = build_dgl_graph(circuit_B_gates, self.num_qubits)
        
        return (g1, types1, locs1), (g2, types2, locs2)

def collate_graphs(samples):
    """
    Batches graphs for DGL.
    Input: List of pairs ((g1, t1, l1), (g2, t2, l2))
    Output: (Batched_G1, T1, L1), (Batched_G2, T2, L2)
    """
    batch_A = [s[0] for s in samples]
    batch_B = [s[1] for s in samples]
    
    def process_batch(batch_data):
        graphs, types, locs = zip(*batch_data)
        batched_graph = dgl.batch(graphs)
        batched_types = torch.cat(types)
        batched_locs = torch.cat(locs)
        return batched_graph, batched_types, batched_locs
        
    return process_batch(batch_A), process_batch(batch_B)

# ==========================================
# 4. Training Loop
# ==========================================
def train():
    # Settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 5
    
    # Data
    dataset = SyntheticQuantumDataset(size=500)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_graphs)
    
    # Model
    model = FidelityEncoder(num_gate_types=10, max_qubits=10).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = CircuitInfoNCELoss(temperature=0.07).to(DEVICE)
    
    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        
        for batch_idx, (batch_A, batch_B) in enumerate(loader):
            # Unpack Batch A
            g1, t1, l1 = batch_A
            g1, t1, l1 = g1.to(DEVICE), t1.to(DEVICE), l1.to(DEVICE)
            
            # Unpack Batch B
            g2, t2, l2 = batch_B
            g2, t2, l2 = g2.to(DEVICE), t2.to(DEVICE), l2.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Passes
            z1 = model(g1, t1, l1) # Embeddings for Circuit A
            z2 = model(g2, t2, l2) # Embeddings for Circuit B
            
            # Calculate Loss
            # This forces z1 and z2 to be close, and far from other batch items
            loss = criterion(z1, z2)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
        
    # ==========================================
    # 5. Verification
    # ==========================================
    print("\n--- Verification ---")
    model.eval()
    with torch.no_grad():
        # Test on the last batch
        # Cosine similarity between Positive Pairs
        cos_sim = F.cosine_similarity(z1, z2)
        print(f"Average Cosine Similarity of Equivalent Circuits: {cos_sim.mean().item():.4f}")
        print("(Should be close to 1.0)")
        
        # Cosine similarity between Negative Pairs (random shift)
        z2_shifted = torch.roll(z2, shifts=1, dims=0)
        cos_sim_neg = F.cosine_similarity(z1, z2_shifted)
        print(f"Average Cosine Similarity of Random Pairs: {cos_sim_neg.mean().item():.4f}")
        print("(Should be close to 0.0)")

    # Save the encoder for use in LayerDAG
    torch.save(model.state_dict(), "fidelity_encoder.pth")
    print("Model saved to fidelity_encoder.pth")

if __name__ == "__main__":
    train()