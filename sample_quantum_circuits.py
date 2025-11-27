import os
import torch
import sys
sys.path.append('src')

from pprint import pprint
from tqdm import tqdm
from setup_utils import set_seed
from src.dataset import load_dataset, DAGDataset
from src.model import DiscreteDiffusion, EdgeDiscreteDiffusion, LayerDAG

def sample_quantum_circuits(args, device, num_categories, model, subset, subset_name):
    """
    Sample quantum circuits conditioned on labels from a dataset subset.

    Args:
        args: Command line arguments
        device: torch device (cuda or cpu)
        num_categories: Number of categories per dimension from dataset
        model: Trained LayerDAG model
        subset: Dataset subset (val_set or test_set)
        subset_name: Name of subset ('validation' or 'test')

    Returns:
        syn_set: DAGDataset containing generated circuits
    """
    syn_set = DAGDataset(num_categories, label=True, multi_dim_features=True)

    print(f"\n{'='*60}")
    print(f"Sampling {len(subset)} circuits for {subset_name} set")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}\n")

    raw_y_batch = []
    num_generated = 0

    for i, y in enumerate(tqdm(subset.y, desc=f"Generating {subset_name} circuits")):
        raw_y_batch.append(y)

        # Generate when batch is full or at the end
        if (len(raw_y_batch) == args.batch_size) or (i == len(subset.y) - 1):
            # Sample from the model
            batch_edge_index, batch_x_n, batch_y = model.sample(
                device,
                len(raw_y_batch),
                raw_y_batch,
                min_num_steps_n=args.min_num_steps_n,
                max_num_steps_n=args.max_num_steps_n,
                min_num_steps_e=args.min_num_steps_e,
                max_num_steps_e=args.max_num_steps_e
            )

            # Add each generated circuit to the synthetic dataset
            for j in range(len(batch_edge_index)):
                edge_index_j = batch_edge_index[j]
                dst_j, src_j = edge_index_j.cpu()
                syn_set.add_data(src_j, dst_j, batch_x_n[j].cpu(), batch_y[j])
                num_generated += 1

            raw_y_batch = []

    print(f"\n✓ Generated {num_generated} circuits for {subset_name} set")
    return syn_set

def save_to_file(syn_set, file_name, sample_dir):
    """Save generated circuits to a .pth file."""
    file_path = os.path.join(sample_dir, file_name)

    data_dict = {
        'src_list': [],
        'dst_list': [],
        'x_n_list': [],
        'y_list': []
    }

    for i in range(len(syn_set)):
        src_i, dst_i, x_n_i, y_i = syn_set[i]
        data_dict['src_list'].append(src_i)
        data_dict['dst_list'].append(dst_i)
        data_dict['x_n_list'].append(x_n_i)
        data_dict['y_list'].append(y_i)

    torch.save(data_dict, file_path)
    print(f"✓ Saved to: {file_path}")

    # Print statistics
    num_nodes = [len(x_n) for x_n in data_dict['x_n_list']]
    num_edges = [len(src) for src in data_dict['src_list']]

    print(f"  Statistics:")
    print(f"    - Num circuits: {len(syn_set)}")
    print(f"    - Avg nodes: {sum(num_nodes)/len(num_nodes):.1f}")
    print(f"    - Avg edges: {sum(num_edges)/len(num_edges):.1f}")
    print(f"    - Node range: [{min(num_nodes)}, {max(num_nodes)}]")
    print(f"    - Edge range: [{min(num_edges)}, {max(num_edges)}]")

def main(args):
    # Set random seed
    set_seed(args.seed)

    # Set number of threads
    torch.set_num_threads(args.num_threads)

    # Setup device
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"\nUsing device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)

    dataset = ckpt['dataset']
    assert dataset == 'quantum_circuits', f"Expected quantum_circuits, got {dataset}"

    print(f"Dataset: {dataset}")
    print(f"\nModel configuration:")
    pprint(ckpt['model_config'])

    # Create diffusion models and move to device
    node_diffusion = DiscreteDiffusion(**ckpt['node_diffusion_config']).to(device)
    edge_diffusion = EdgeDiscreteDiffusion(**ckpt['edge_diffusion_config']).to(device)

    # Create model
    model = LayerDAG(
        device=device,
        node_diffusion=node_diffusion,
        edge_diffusion=edge_diffusion,
        **ckpt['model_config']
    )

    # Load trained weights and move to device
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")

    # Load datasets
    print(f"\nLoading quantum circuits dataset...")
    train_set, val_set, test_set = load_dataset('quantum_circuits')
    print(f"  Train: {len(train_set)} circuits")
    print(f"  Val: {len(val_set)} circuits")
    print(f"  Test: {len(test_set)} circuits")

    # Create output directory
    sample_dir = args.output_dir
    os.makedirs(sample_dir, exist_ok=True)
    print(f"\nOutput directory: {sample_dir}")

    # Sample validation set if requested
    if args.sample_val:
        val_syn_set = sample_quantum_circuits(
            args, device, train_set.num_categories, model, val_set, 'validation'
        )
        save_to_file(val_syn_set, 'validation.pth', sample_dir)

    # Sample test set if requested
    if args.sample_test:
        test_syn_set = sample_quantum_circuits(
            args, device, train_set.num_categories, model, test_set, 'test'
        )
        save_to_file(test_syn_set, 'test.pth', sample_dir)

    # Sample training set if requested (useful for distribution comparison)
    if args.sample_train:
        train_syn_set = sample_quantum_circuits(
            args, device, train_set.num_categories, model, train_set, 'training'
        )
        save_to_file(train_syn_set, 'train.pth', sample_dir)

    print(f"\n{'='*60}")
    print("✓ Sampling complete!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Sample quantum circuits from trained LayerDAG model")

    # Model and paths
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint (.pth file)")
    parser.add_argument("--output_dir", type=str, default="quantum_circuits_samples",
                        help="Directory to save generated circuits")

    # Sampling configuration
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation")
    parser.add_argument("--sample_val", action="store_true", default=True,
                        help="Sample validation set")
    parser.add_argument("--sample_test", action="store_true", default=True,
                        help="Sample test set")
    parser.add_argument("--sample_train", action="store_false",
                        help="Sample training set (default: False)")

    # Diffusion sampling parameters
    parser.add_argument("--min_num_steps_n", type=int, default=None,
                        help="Minimum number of diffusion steps for nodes")
    parser.add_argument("--max_num_steps_n", type=int, default=None,
                        help="Maximum number of diffusion steps for nodes")
    parser.add_argument("--min_num_steps_e", type=int, default=None,
                        help="Minimum number of diffusion steps for edges")
    parser.add_argument("--max_num_steps_e", type=int, default=None,
                        help="Maximum number of diffusion steps for edges")

    # System configuration
    parser.add_argument("--num_threads", type=int, default=16,
                        help="Number of CPU threads")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")

    args = parser.parse_args()

    main(args)
