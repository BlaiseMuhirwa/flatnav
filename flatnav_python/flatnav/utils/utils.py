from typing import List
import os


def load_from_mtx_file(filename: str) -> List[List[int]]:
    """
    Load an adjacency list from a Matrix Market-formatted file.
    :param filename: Path to the input file
    :return: The adjacency list
    
    """
    
    if not filename.endswith(".mtx"):
        raise ValueError(
            "Input file must be in Matrix Market format with the extension .mtx"
        )

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")

    with open(filename, "r") as input_file:
        # Skip header lines starting with '%'
        line = next(input_file)
        while line.startswith("%"):
            line = next(input_file)

        # Read the dimensions
        num_vertices, _, num_edges = map(int, line.split())

        # Initialize adjacency list
        adjacency_list = [[] for _ in range(num_vertices)]

        # Read edges
        for line in input_file:
            u, v = map(int, line.split())
            # Adjust for 1-based indexing in Matrix Market format
            adjacency_list[u - 1].append(v - 1)

    return adjacency_list
