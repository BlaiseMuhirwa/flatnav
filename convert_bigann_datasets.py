import numpy as np
import os
import sys

def load_ground_truth(path):
    with open(path, "rb") as f:
        num_queries = np.fromfile(f, dtype=np.uint32, count=1)[0]
        k = np.fromfile(f, dtype=np.uint32, count=1)[0]

    ground_truth_ids = np.memmap(path, dtype=np.uint32, mode="r", shape=(num_queries, k), offset=8,)
    
    np.save(f'data/bigann/bigann_ground_truth', ground_truth_ids)


def load_bigann_vectors(path, queries=False):
    dtype = np.float32 if path.endswith("fbin") else np.uint8

    # Read header information (num_points and num_dimensions)
    with open(path, "rb") as f:
        num_queries = np.fromfile(f, dtype=np.uint32, count=1)[0]
        num_dimensions = np.fromfile(f, dtype=np.uint32, count=1)[0]

    dataset = np.fromfile(path, dtype=dtype, offset=8)
    dataset = dataset.reshape((num_queries, num_dimensions))

    if queries:
        np.save(f'data/bigann/bigann_queries', dataset)
    else:
        np.save(f'data/bigann/bigann_train', dataset)


path = sys.argv[1]
mode = sys.argv[2]

if mode == "train":
    load_bigann_vectors(path)

elif mode == "queries":
    load_bigann_vectors(path, True)

elif mode == "gt":
    load_ground_truth(path)

