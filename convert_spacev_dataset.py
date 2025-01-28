"""
https://github.com/microsoft/SPTAG/blob/main/datasets/SPACEV1B/README.md
"""
import struct
import numpy as np
import os
import sys

def load_spacev_vectors(path):
    part_count = len(os.listdir(path))

    for i in range(1, part_count + 1):
        fvec = open(os.path.join(path, 'vectors_%d.bin' % i), 'rb')
        if i == 1:
            vec_count = struct.unpack('i', fvec.read(4))[0]
            vec_dimension = struct.unpack('i', fvec.read(4))[0]
            vecbuf = bytearray(vec_count * vec_dimension)
            vecbuf_offset = 0
        while True:
            part = fvec.read(1048576)
            if len(part) == 0: break
            vecbuf[vecbuf_offset: vecbuf_offset + len(part)] = part
            vecbuf_offset += len(part)
        fvec.close()
    
    base_path, _ = os.path.split(path)
    collection = np.frombuffer(vecbuf, dtype=np.int8).reshape((vec_count, vec_dimension))
    
    collection = collection[:100000000]
    np.save(os.path.join(base_path, 'train_100m'), collection)

    collection = collection[:10000000]
    np.save(os.path.join(base_path, 'train_10m'), collection)


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
