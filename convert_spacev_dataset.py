"""
https://github.com/microsoft/SPTAG/blob/main/datasets/SPACEV1B/README.md
"""
import struct
import numpy as np
import os
import sys


def load_spacev_vectors(path):
    part_count = len(os.listdir(path))

    # The 8-byte header (vec_count, vec_dimension) lives only in part 1; the
    # remaining bytes of part 1 and all of parts 2..N are raw vector data
    # concatenated in order.
    with open(os.path.join(path, 'vectors_1.bin'), 'rb') as fvec:
        vec_count = struct.unpack('i', fvec.read(4))[0]
        vec_dimension = struct.unpack('i', fvec.read(4))[0]

    collection = np.empty((vec_count, vec_dimension), dtype=np.int8)
    flat = collection.reshape(-1)
    needed = flat.shape[0]
    filled = 0

    for i in range(1, part_count + 1):
        if filled >= needed:
            break
        with open(os.path.join(path, 'vectors_%d.bin' % i), 'rb') as fvec:
            if i == 1:
                # skip the header on the first part only
                fvec.seek(8)               
            while filled < needed:
                chunk = fvec.read(min(1048576, needed - filled))
                if len(chunk) == 0:
                    break
                buf = np.frombuffer(chunk, dtype=np.int8)
                flat[filled: filled + buf.shape[0]] = buf
                filled += buf.shape[0]

    if filled != needed:
        raise ValueError(
            f"Expected {needed} bytes for {vec_count} vectors but only read {filled}"
        )

    # Slicing clamps automatically if the file holds fewer rows than the cutoff.
    base_path, _ = os.path.split(path)
    np.save(os.path.join(base_path, 'train_100m'), collection[:100000000])
    np.save(os.path.join(base_path, 'train_10m'), collection[:10000000])


def load_spacev_queries(path):
    fq = open(path, 'rb')
    q_count = struct.unpack('i', fq.read(4))[0]
    q_dimension = struct.unpack('i', fq.read(4))[0]
    queries = np.frombuffer(fq.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))

    base_path, _ = os.path.split(path)
    np.save(os.path.join(base_path, "queries"), queries)

path = sys.argv[1]

if not os.path.exists(path):
    raise ValueError(f"The provided path {path} does not exist")

mode = sys.argv[2]

if mode == "train":
    load_spacev_vectors(path)
elif mode == "queries":
    load_spacev_queries(path)
else:
    raise ValueError(f"Input mode: {mode} not recognized")
