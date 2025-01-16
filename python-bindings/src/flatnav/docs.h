#pragma once 

// One sad thing about this is that the docstrings are likely to become stale
// as the code evolves. Nonetheless, it's good to have them in one place.


static const char *ADD_DOCSTRING = R"pbdoc(
Add vectors(data) to the index with the given `ef_construction` parameter and optional labels. 
`ef_construction` determines how many vertices are visited while inserting every vector in 
the underlying graph structure.
Args:
    data (np.ndarray): The data to add to the index.
    ef_construction (int): The number of vertices to visit while inserting every vector in the graph.
    num_initializations (int, optional): The number of initializations to perform. Defaults to 100.
    labels (Optional[np.ndarray], optional): The labels for the data. Defaults to None.
Returns:
    None
)pbdoc";

static const char *ALLOCATE_NODES_DOCSTRING = R"pbdoc(
Allocate nodes in the underlying graph structure for the given data. Unlike the add method, 
this method does not construct the edge connectivity. It only allocates memory for each node 
in the graph. When using this method, you should invoke `build_graph_links` explicity. 
```NOTE```: In most cases you should not need to use this method.
Args:
    data (np.ndarray): The data to add to the index.
Returns:
    None
)pbdoc";

static const char *SEARCH_SINGLE_DOCSTRING = R"pbdoc(
Return top `K` closest data points for the given `query`. The results are returned as a Tuple of 
distances and label ID's. The `ef_search` parameter determines how many neighbors are visited 
while finding the closest neighbors for the query.

Args:
    query (np.ndarray): The query vector.
    K (int): The number of neighbors to return.
    ef_search (int): The number of neighbors to visit while finding the closest neighbors for the query.
    num_initializations (int, optional): The number of initializations to perform. Defaults to 100.
Returns:
    Tuple[np.ndarray, np.ndarray]: The distances and label ID's of the closest neighbors.
)pbdoc";

static const char *SEARCH_DOCSTRING = R"pbdoc(
This is a batched version of the `search_single` method.
Return top `K` closest data points for every query in the provided `queries`. The results are returned as a Tuple of
distances and label ID's. The `ef_search` parameter determines how many neighbors are visited while finding the closest neighbors
for every query.

Args:
    queries (np.ndarray): The query vectors.
    K (int): The number of neighbors to return.
    ef_search (int): The number of neighbors to visit while finding the closest neighbors for every query.
    num_initializations (int, optional): The number of initializations to perform. Defaults to 100.
Returns:
    Tuple[np.ndarray, np.ndarray]: The distances and label ID's of the closest neighbors.
)pbdoc";

static const char *GET_GRAPH_OUTDEGREE_TABLE_DOCSTRING = R"pbdoc(
Returns the outdegree table (adjacency list) representation of the underlying graph.
Returns:
    List[List[int]]: The outdegree table.
)pbdoc";

static const char *BUILD_GRAPH_LINKS_DOCSTRING = R"pbdoc(
Construct the edge connectivity of the underlying graph. This method should be invoked after 
allocating nodes using the `allocate_nodes` method.
Args:
    mtx_filename (str): The filename of the matrix file.

Returns:
    None
)pbdoc";

static const char *REORDER_DOCSTRING = R"pbdoc(
Perform graph re-ordering based on the given sequence of re-ordering strategies.
Supported re-ordering strategies include `gorder` and `rcm`.
Reference: 
  1. Graph Reordering for Cache-Efficient Near Neighbor Search: https://arxiv.org/pdf/2104.03221
Args:
    strategies (List[str]): The sequence of re-ordering strategies.
Returns:
    None
)pbdoc";

static const char *SET_NUM_THREADS_DOCSTRING = R"pbdoc(
Set the number of threads to use for constructing the graph and/or performing KNN search.
Args:
    num_threads (int): The number of threads to use.
Returns:
    None
)pbdoc";

static const char *NUM_THREADS_DOCSTRING = R"pbdoc(
Returns the number of threads used for constructing the graph and/or performing KNN search.
Returns:
    int: The number of threads.
)pbdoc";

static const char *MAX_EDGES_PER_NODE_DOCSTRING = R"pbdoc(
Maximum number of edges(links) per node in the underlying NSW graph data structure.
Returns:
    int: The maximum number of edges per node.
)pbdoc";

static const char *SAVE_DOCSTRING = R"pbdoc(
Save a FlatNav index at the given file location.
Args:
    filename (str): The file location to save the index.
Returns:
    None
)pbdoc";

static const char *LOAD_INDEX_DOCSTRING = R"pbdoc(
Load a FlatNav index from a given file location.
Args:
    filename (str): The file location to load the index from.
Returns:
    Union[L2Inde, IndexIPFloat]: The loaded index.
)pbdoc";

static const char *GET_QUERY_DISTANCE_COMPUTATIONS_DOCSTRING = R"pbdoc(
Returns the number of distance computations performed during the last search operation. 
This method also resets the distance computations counter.
Returns:
    int: The number of distance computations.
)pbdoc";

static const char *CONSTRUCTOR_DOCSTRING = R"pbdoc(
Constructs a an in-memory index with the parameters.
Args:
    distance_type (str): The type of distance metric to use ('l2' for Euclidean, 'angular' for inner product).
    dim (int): The number of dimensions in the dataset.
    dataset_size (int): The number of vectors in the dataset.
    max_edges_per_node (int): The maximum number of edges per node in the graph.
    verbose (bool, optional): Enables verbose output. Defaults to False.
    collect_stats (bool, optional): Collects performance statistics. Defaults to False.

Returns:
    Union[IndexL2Float, IndexIPFloat]: The constructed index.
)pbdoc";


static const char* CXX_EXTENSION_MODULE_DOCSTRING = R"pbdoc(
Flatnav: A performant graph-based kNN search library with re-ordering
======================================================================

Provides:
    - An efficient graph-based approach to kNN search.
    - Re-ordering capabilities for cache efficiency.
    - Comprehensive utilities for index construction and query.

Documentation:
    https://flatnav.net

Source Code:
    https://github.com/BlaiseMuhirwa/flatnav

Submodules:
    data_type:
        Definitions of supported data types for the index (e.g., float32, int8).
    index:
        Methods and classes for constructing, querying, and managing indices.

Example:
    ```python
    import flatnav
    from flatnav import DataType

    # Create an index
    index = flatnav.index.create(
        distance_type='l2',
        dim=128,
        dataset_size=10000,
        max_edges_per_node=32,
    )

    # Add data to the index
    data = np.random.rand(100, 128).astype(np.float32)
    index.add(data, ef_construction=100)

    # Perform a search
    query = np.random.rand(128).astype(np.float32)
    distances, labels = index.search_single(query, K=10, ef_search=50)
    print("Nearest neighbors:", labels)
    ```

Utilities:
    - DataType: Supported data types for the index.
    - MetricType: Supported distance metrics (L2, Inner Product).
    )pbdoc";