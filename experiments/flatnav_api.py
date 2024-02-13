import flatnav
from flatnav.index import index_factory, IndexBuilder
import numpy as np

index_builder = (
    IndexBuilder(distance_type="l2", dataset_size=1000, dim=12)
    .with_index_params(
        {
            "max-edges-per-node": 16,
            "efConstruction": 200,
            "num-initializations": 100,
            "num-threads": 4,
        }
    )
    .with_index_name("hnsw_16_200")
)

index = index_factory(builder=index_builder)
index.add(data=np.random.rand(1000, 12))
index_builder.with_reordering(["gorder", "rcm"])

index.search(queries=np.random.rand(100, 12), k=10)

##########################################
# second initialization via a MTX file
##########################################

outdegree_table = index.utils.load_from_mtx_file(filename="outdegree.mtx")

# this calls buildGraphLinks() internally
index = index_factory(builder=index_builder, outdegree_table=outdegree_table)
