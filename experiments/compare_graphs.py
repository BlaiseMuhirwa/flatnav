import numpy as np 
import os


def write_adjacency_lists_to_txt(flatnav_graph: np.ndarray, hnsw_graph: np.ndarray) -> None:
    flatnav_file = "/root/data/flatnav_table.txt"
    hnsw_file = "/root/data/hnsw_table.txt"

    with open(flatnav_file, 'w') as f_flatnav:
        for row in flatnav_graph:
            f_flatnav.write(" ".join(map(str, row)) + "\n")

    with open(hnsw_file, 'w') as f_hnsw:
        for row in hnsw_graph:
            f_hnsw.write(" ".join(map(str, row)) + "\n")

    print(f"Adjacency lists written to {flatnav_file} and {hnsw_file}")


def compare_graphs(flatnav_path: str, hnsw_path: str) -> None:
    if not os.path.exists(flatnav_path):
        raise ValueError(f"Invalid Path: {flatnav_path}")

    if not os.path.exists(hnsw_path):
        raise ValueError(f"Invalid Path: {hnsw_path}")

    flatnav_graph = np.load(flatnav_path, allow_pickle=True)
    hnsw_graph = np.load(hnsw_path, allow_pickle=True)

    write_adjacency_lists_to_txt(flatnav_graph=flatnav_graph, hnsw_graph=hnsw_graph)

    assert len(flatnav_graph) == len(hnsw_graph), "Invalid sizes for the adjacency lists"

    size = len(flatnav_graph)

    for i in range(size):
        edge_set_len_flatnav = len(flatnav_graph[i])
        edge_set_len_hnsw = len(hnsw_graph[i])

        if edge_set_len_flatnav != edge_set_len_hnsw:
            print(f"Value of i: {i}")
            raise RuntimeError(f"Edge set length for flatnav: {edge_set_len_flatnav}. Edge set length for hnsw: {edge_set_len_hnsw}")

        flatnav_set = set(flatnav_graph[i])
        hnsw_set = set(hnsw_graph[i])

        if flatnav_set != hnsw_set:
            print(f"FlatNav Length: {len(flatnav_set)}. HNSW Length: {len(hnsw_set)}")
            raise RuntimeError(f"Edge sets not equal. i={i}")
        # for j in range(edge_set_len_flatnav):
        #     if flatnav_graph[i][j] != hnsw_graph[i][j]:
        #         raise RuntimeError(f"flatnav_graph[{i}][{j}]={flatnav_graph[i][j]}. hnsw_graph[{i}][{j}]={hnsw_graph[i][j]}")


    print("No Difference Found")

if __name__=="__main__":
    compare_graphs(
        flatnav_path="/root/data/outdegree_table_flatnav.npy",
        hnsw_path="/root/data/outdegree_table_hnsw.npy"
    )


# Flanav for node 1465
# 722 429 1198 4674 187 982 1101 2258 1813 553 1450 4564 3781 778 1066 6757 8508 8914 8983 9367 9963 11464 11673 11756 14935

# HNSW for node 1465
# 722 429 1198 187 3411 982 1101 2258 1813 553 3825 4564 3781 778 1066 6757 8508 8914 8983 9367 9963 11464 11673 11756 14935