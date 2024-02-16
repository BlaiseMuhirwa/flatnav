import flatnav
from flatnav.index import create_index, IndexBuilder
import numpy as np

index_builder = (
    IndexBuilder(dataset_size=10000)
    .with_index_params(
        {
            "max_edges_per_node": 16,
            "ef_construction": 200,
            "num_initializations": 100,
            "num_threads": 4,
            "index_name": "hnsw_16_200"
        }
    )
    .with_reordering(["gorder", "rcm"]) 
)

print(f"Index builder: {index_builder}")

index = create_index(distance_type="l2", dim=100, index_builder=index_builder)

print(f"Index: {index}")
index.add(data=np.random.rand(10000, 100))

threads = index.num_threads
print(f"Number of threads: {threads}")

index.set_num_threads(8)

threads = index.num_threads
print(f"Number of threads: {threads}")
# index_builder.with_reordering(["gorder", "rcm"])

# index.search(queries=np.random.rand(100, 12), k=10)

# ##########################################
# # second initialization via a MTX file
# ##########################################

outdegree_table = flatnav.utils.load_from_mtx_file(filename="outdegree.mtx")

# this calls buildGraphLinks() internally
index.build_from_outdegree_table(outdegree_table)
