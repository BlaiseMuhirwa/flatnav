import numpy as np
import os

def load_bigann_ground_truth(path):
    with open(path, "rb") as f:
        num_queries = np.fromfile(f, dtype=np.uint32, count=1)[0]
        k = np.fromfile(f, dtype=np.uint32, count=1)[0]

    ground_truth_ids = np.memmap(path, dtype=np.uint32, mode="r", shape=(num_queries, k), offset=8,)
    return ground_truth_ids


def load_bigann_vectors(path):
    dtype = np.float32 if path.endswith("fbin") else np.uint8

    # Read header information (num_points and num_dimensions)
    with open(path, "rb") as f:
        num_queries = np.fromfile(f, dtype=np.uint32, count=1)[0]
        num_dimensions = np.fromfile(f, dtype=np.uint32, count=1)[0]

    dataset = np.fromfile(path, dtype=dtype, offset=8)
    dataset = dataset.reshape((num_queries, num_dimensions))

    np.save(f'bigann_queries', dataset)



load_bigann_vectors("/Users/vihan/Documents/flatnav/data/bigann_query.public.10K.u8bin")
