import numpy as np
import os
import argparse


def _size_label(n):
    """Convert a number to a human-readable label (e.g., 10000000 -> '10m', 1000000000 -> '1b')."""
    if n >= 1_000_000_000 and n % 1_000_000_000 == 0:
        return f'{n // 1_000_000_000}b'
    elif n >= 1_000_000 and n % 1_000_000 == 0:
        return f'{n // 1_000_000}m'
    elif n >= 1_000 and n % 1_000 == 0:
        return f'{n // 1_000}k'
    else:
        return str(n)


def load_ground_truth(path, output_dir=None):
    with open(path, "rb") as f:
        num_queries = np.fromfile(f, dtype=np.uint32, count=1)[0]
        k = np.fromfile(f, dtype=np.uint32, count=1)[0]

    ground_truth_ids = np.memmap(path, dtype=np.uint32, mode="r", shape=(num_queries, k), offset=8,)
    _, gt_filename = os.path.split(path)

    dataset_name, size = gt_filename.split('-')
    if output_dir is None:
        save_dir = os.path.join("data", dataset_name)
    else:
        save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'ground_truth_{size.lower()}'), ground_truth_ids)


def load_bigann_vectors(path, queries=False, chunk_size=None):
    dtype = np.float32 if path.endswith("fbin") else np.uint8

    # Read header information (num_points and num_dimensions)
    with open(path, "rb") as f:
        num_items = np.fromfile(f, dtype=np.uint32, count=1)[0]
        num_dimensions = np.fromfile(f, dtype=np.uint32, count=1)[0]

    dataset = np.fromfile(path, dtype=dtype, offset=8)
    dataset = dataset.reshape((num_items, num_dimensions))

    base_path, _ = os.path.split(path)

    if queries:
        np.save(os.path.join(base_path, 'queries'), dataset)
    else:
        if chunk_size is not None:
            if chunk_size > num_items:
                print(f"Warning: chunk_size {chunk_size} > dataset size {num_items}. Saving full dataset.")
                np.save(os.path.join(base_path, 'train'), dataset)
            else:
                label = _size_label(chunk_size)
                np.save(os.path.join(base_path, f'train_{label}'), dataset[:chunk_size])
        else:
            np.save(os.path.join(base_path, 'train'), dataset)


parser = argparse.ArgumentParser(description="Convert BigANN binary datasets to .npy format")
parser.add_argument("path", help="Path to the input binary file")
parser.add_argument("mode", choices=["train", "queries", "gt"], help="Conversion mode")
parser.add_argument("--output-dir", default=None, help="Output directory (for gt mode)")
parser.add_argument("--chunk-size", type=int, default=None,
                    help="Number of train vectors to keep (default: save all)")
args = parser.parse_args()


if args.mode == "train":
    load_bigann_vectors(args.path, chunk_size=args.chunk_size)
elif args.mode == "queries":
    load_bigann_vectors(args.path, True)
elif args.mode == "gt":
    load_ground_truth(args.path, args.output_dir)
else:
    raise ValueError(f"Input mode: {args.mode} not recognized")
