"""
https://github.com/microsoft/SPTAG/blob/main/datasets/SPACEV1B/README.md
"""
import struct
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


def load_spacev_vectors(path, chunk_size=None):
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

    if chunk_size is not None:
        if chunk_size > vec_count:
            print(f"Warning: chunk_size {chunk_size} > dataset size {vec_count}. Saving full dataset.")
            np.save(os.path.join(base_path, 'train'), collection)
        else:
            label = _size_label(chunk_size)
            np.save(os.path.join(base_path, f'train_{label}'), collection[:chunk_size])
    else:
        np.save(os.path.join(base_path, 'train'), collection)


def load_spacev_queries(path):
    fq = open(path, 'rb')
    q_count = struct.unpack('i', fq.read(4))[0]
    q_dimension = struct.unpack('i', fq.read(4))[0]
    queries = np.frombuffer(fq.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))

    base_path, _ = os.path.split(path)
    np.save(os.path.join(base_path, "queries"), queries)


parser = argparse.ArgumentParser(description="Convert SPACEV binary datasets to .npy format")
parser.add_argument("path", help="Path to the input binary file")
parser.add_argument("mode", choices=["train", "queries"], help="Conversion mode")
parser.add_argument("--chunk-size", type=int, default=None,
                    help="Number of train vectors to keep (default: save all)")
args = parser.parse_args()

if not os.path.exists(args.path):
    raise ValueError(f"The provided path {args.path} does not exist")

if args.mode == "train":
    load_spacev_vectors(args.path, chunk_size=args.chunk_size)
elif args.mode == "queries":
    load_spacev_queries(args.path)
else:
    raise ValueError(f"Input mode: {args.mode} not recognized")
