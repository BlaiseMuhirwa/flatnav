import numpy as np
import os
import sys

def load_ground_truth(path):
    with open(path, "rb") as f:
        num_queries = np.fromfile(f, dtype=np.uint32, count=1)[0]
        k = np.fromfile(f, dtype=np.uint32, count=1)[0]

    ground_truth_ids = np.memmap(path, dtype=np.uint32, mode="r", shape=(num_queries, k), offset=8,)
    _, gt_filename = os.path.split(path)
    
    dataset_name, size = gt_filename.split('-')
    data_dir = os.environ.get("DATA_DIR", "data")
    save_dir = os.path.join(data_dir, dataset_name)
    np.save(os.path.join(save_dir, f'ground_truth_{size.lower()}'), ground_truth_ids)


def load_bigann_vectors(path, queries=False):
    dtype = np.float32 if path.endswith("fbin") else np.uint8

    # Read header information (num_points and num_dimensions)
    with open(path, "rb") as f:
        num_items = np.fromfile(f, dtype=np.uint32, count=1)[0]
        num_dimensions = np.fromfile(f, dtype=np.uint32, count=1)[0]

    dataset = np.memmap(path, dtype=dtype, mode="r", offset=8,
                        shape=(int(num_items), int(num_dimensions)))

    base_path, _ = os.path.split(path)

    if queries:
        np.save(os.path.join(base_path, 'queries'), dataset)
    else:
        # Slicing a memmap is lazy: only the first 100M / 10M rows are read
        # during np.save, so the remaining ~900M rows are never touched. Slices
        # clamp automatically if the file holds fewer rows than the cutoff.
        dataset_100m = dataset[:100000000]
        dataset_10m = dataset[:10000000]

        np.save(os.path.join(base_path, 'train_100m'), dataset_100m)
        np.save(os.path.join(base_path, 'train_10m'), dataset_10m)


path = sys.argv[1]
mode = sys.argv[2]


if mode == "train":
    load_bigann_vectors(path)
elif mode == "queries":
    load_bigann_vectors(path, True)
elif mode == "gt":
    load_ground_truth(path)
else:
    raise ValueError(f"Input mode: {mode} not recognized")
