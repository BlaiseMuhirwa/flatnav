import numpy as np
import os

def load_ground_truth(path):

    with open(path, "rb") as f:
        num_queries = np.fromfile(f, dtype=np.uint32, count=1)[0]
        K = np.fromfile(f, dtype=np.uint32, count=1)[0]
        print(num_queries, K)

    ground_truth_ids = np.memmap(path, dtype=np.uint32, mode="r", shape=(num_queries, K), offset=8,)

    return ground_truth_ids


def load_bigann_vectors(path):

    dtype = np.float32 if path.endswith("fbin") else np.uint8

    # Read header information (num_points and num_dimensions)
    with open(path, "rb") as f:
        num_queries = np.fromfile(f, dtype=np.uint32, count=1)[0]
        num_dimensions = np.fromfile(f, dtype=np.uint32, count=1)[0]

    queries_dataset = np.fromfile(
            path,
            dtype=dtype,
            offset=8,)

    queries_dataset = queries_dataset.reshape((num_queries, num_dimensions))
    print(np.shape(queries_dataset))
    np.save(f'bigann_queries', queries_dataset)



load_bigann_vectors("/Users/vihan/Documents/flatnav/data/bigann_query.public.10K.u8bin")
