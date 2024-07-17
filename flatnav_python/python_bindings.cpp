#include <algorithm>
#include <flatnav/distance_interface.h>
#include <flatnav/index.h>
#include <flatnav/distances/inner_product_distance.h>
#include <flatnav/distances/squared_l2_distance.h>
#include <flatnav/util/parallel_constructs.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using flatnav::DistanceInterface;
using flatnav::Index;
using flatnav::InnerProductDistance;
using flatnav::SquaredL2Distance;

namespace py = pybind11;

template <typename dist_t, typename label_t>
class PyIndex : public std::enable_shared_from_this<PyIndex<dist_t, label_t>> {
  const uint32_t NUM_LOG_STEPS = 10000;

private:
  int _dim;
  label_t _label_id;
  bool _verbose;
  Index<dist_t, label_t> *_index;

public:
  typedef std::pair<py::array_t<float>, py::array_t<label_t>>
      DistancesLabelsPair;

  explicit PyIndex(std::unique_ptr<Index<dist_t, label_t>> index)
      : _dim(index->dataDimension()), _label_id(0), _verbose(false),
        _index(index.release()) {

    if (_verbose) {
      _index->getIndexSummary();
    }
  }

  PyIndex(std::unique_ptr<DistanceInterface<dist_t>> &&distance,
          int dataset_size, int max_edges_per_node, bool verbose = false,
          bool collect_stats = false)
      : _dim(distance->dimension()), _label_id(0), _verbose(verbose),
        _index(new Index<dist_t, label_t>(
            /* dist = */ std::move(distance),
            /* dataset_size = */ dataset_size,
            /* max_edges_per_node = */ max_edges_per_node,
            /* collect_stats = */ collect_stats)) {

    if (_verbose) {
      uint64_t total_index_memory = _index->getTotalIndexMemory();
      uint64_t visited_set_allocated_memory =
          _index->visitedSetPoolAllocatedMemory();
      uint64_t mutexes_allocated_memory = _index->mutexesAllocatedMemory();

      auto total_memory = total_index_memory + visited_set_allocated_memory +
                          mutexes_allocated_memory;

      std::cout << "Total allocated index memory: " << total_memory / 1e9
                << " GB \n"
                << std::flush;
      std::cout << "[WARN]: More memory might be allocated due to visited sets "
                   "in multi-threaded environments.\n"
                << std::flush;
      _index->getIndexSummary();
    }
  }

  PyIndex(std::unique_ptr<DistanceInterface<dist_t>> &&distance,
          const std::string &mtx_filename, bool verbose = false,
          bool collect_stats = false)
      : _label_id(0), _verbose(verbose),
        _index(
            new Index<dist_t, label_t>(/* dist = */ std::move(distance),
                                       /* mtx_filename = */ mtx_filename,
                                       /* collect_stats = */ collect_stats)) {
    _dim = _index->dataDimension();
  }

  Index<dist_t, label_t> *getIndex() { return _index; }

  ~PyIndex() { delete _index; }

  uint64_t getQueryDistanceComputations() const {
    auto distance_computations = _index->distanceComputations();
    _index->resetStats();
    return distance_computations;
  }

  static std::shared_ptr<PyIndex<dist_t, label_t>>
  loadIndex(const std::string &filename) {
    auto index = Index<dist_t, label_t>::loadIndex(/* filename = */ filename);
    return std::make_shared<PyIndex<dist_t, label_t>>(std::move(index));
  }

  std::shared_ptr<PyIndex<dist_t, label_t>> allocateNodes(
      const py::array_t<float, py::array::c_style | py::array::forcecast>
          &data) {
    auto num_vectors = data.shape(0);
    auto data_dim = data.shape(1);
    if (data.ndim() != 2 || data_dim != _dim) {
      throw std::invalid_argument("Data has incorrect dimensions.");
    }
    for (size_t vec_index = 0; vec_index < num_vectors; vec_index++) {
      uint32_t new_node_id;

      this->_index->allocateNode(/* data = */ (void *)data.data(vec_index),
                                 /* label = */ _label_id,
                                 /* new_node_id = */ new_node_id);
      _label_id++;
    }
    return this->shared_from_this();
  }

  void
  add(const py::array_t<float, py::array::c_style | py::array::forcecast> &data,
      int ef_construction, int num_initializations = 100,
      py::object labels = py::none()) {
    // py::array_t<float, py::array::c_style | py::array::forcecast> means that
    // the functions expects either a Numpy array of floats or a castable type
    // to that type. If the given type can't be casted, pybind11 will throw an
    // error.

    auto num_vectors = data.shape(0);
    auto data_dim = data.shape(1);
    if (data.ndim() != 2 || data_dim != _dim) {
      throw std::invalid_argument(
          "Data has incorrect dimensions. data.ndim() = "
          "`" +
          std::to_string(data.ndim()) + "` and data_dim = `" +
          std::to_string(data_dim) +
          "`. Expected 2D "
          "array with "
          "dimensions "
          "(num_vectors, "
          "dim).");
    }

    if (labels.is_none()) {
      std::vector<label_t> vec_labels(num_vectors);
      std::iota(vec_labels.begin(), vec_labels.end(), 0);

      {
        // Release python GIL while threads are running
        py::gil_scoped_release gil;
        this->_index->addBatch(
            /* data = */ (void *)data.data(0),
            /* labels = */ vec_labels,
            /* ef_construction = */ ef_construction,
            /* num_initializations = */ num_initializations);
      }
      return;
    }

    // Use the provided labels now
    try {
      auto vec_labels = py::cast<std::vector<label_t>>(labels);
      if (vec_labels.size() != num_vectors) {
        throw std::invalid_argument("Incorrect number of labels.");
      }
      {
        // Relase python GIL while threads are running
        py::gil_scoped_release gil;
        this->_index->addBatch(
            /* data = */ (void *)data.data(0), /* labels = */ vec_labels,
            /* ef_construction = */ ef_construction,
            /* num_initializations = */ num_initializations);
      }
    } catch (const py::cast_error &error) {
      throw std::invalid_argument("Invalid labels provided.");
    }
  }

  DistancesLabelsPair searchSingle(
      const py::array_t<float, py::array::c_style | py::array::forcecast>
          &query,
      int K, int ef_search, int num_initializations = 100) {
    if (query.ndim() != 1 || query.shape(0) != _dim) {
      throw std::invalid_argument("Query has incorrect dimensions.");
    }

    std::vector<std::pair<float, label_t>> top_k = this->_index->search(
        /* query = */ (const void *)query.data(0), /* K = */ K,
        /* ef_search = */ ef_search,
        /* num_initializations = */ num_initializations);

    if (top_k.size() != K) {
      throw std::runtime_error(
          "Search did not return the expected number of results. Expected " +
          std::to_string(K) + " but got " + std::to_string(top_k.size()) + ".");
    }

    label_t *labels = new label_t[K];
    float *distances = new float[K];

    for (size_t i = 0; i < K; i++) {
      distances[i] = top_k[i].first;
      labels[i] = top_k[i].second;
    }

    // Allows to transfer ownership to Python
    py::capsule free_labels_when_done(labels,
                                      [](void *ptr) { delete (label_t *)ptr; });

    py::capsule free_distances_when_done(
        distances, [](void *ptr) { delete (float *)ptr; });

    py::array_t<label_t> labels_array = py::array_t<label_t>(
        {K}, {sizeof(label_t)}, labels, free_labels_when_done);

    py::array_t<float> distances_array = py::array_t<float>(
        {K}, {sizeof(float)}, distances, free_distances_when_done);

    return {distances_array, labels_array};
  }

  DistancesLabelsPair
  search(const py::array_t<float, py::array::c_style | py::array::forcecast>
             &queries,
         int K, int ef_search, int num_initializations = 100) {
    size_t num_queries = queries.shape(0);
    size_t queries_dim = queries.shape(1);

    if (queries.ndim() != 2 || queries_dim != _dim) {
      throw std::invalid_argument("Queries have incorrect dimensions.");
    }

    auto num_threads = _index->getNumThreads();
    label_t *results = new label_t[num_queries * K];
    float *distances = new float[num_queries * K];

    // No need to spawn any threads if we are in a single-threaded environment
    if (num_threads == 1) {
      for (size_t query_index = 0; query_index < num_queries; query_index++) {
        std::vector<std::pair<float, label_t>> top_k = this->_index->search(
            /* query = */ (const void *)queries.data(query_index), /* K = */ K,
            /* ef_search = */ ef_search,
            /* num_initializations = */ num_initializations);

        if (top_k.size() != K) {
          throw std::runtime_error("Search did not return the expected number "
                                   "of results. Expected " +
                                   std::to_string(K) + " but got " +
                                   std::to_string(top_k.size()) + ".");
        }

        for (size_t i = 0; i < top_k.size(); i++) {
          distances[query_index * K + i] = top_k[i].first;
          results[query_index * K + i] = top_k[i].second;
        }
      }
    } else {
      // Parallelize the search
      flatnav::executeInParallel(
          /* start_index = */ 0, /* end_index = */ num_queries,
          /* num_threads = */ num_threads,
          /* function = */ [&](uint32_t row_index) {
            auto *query = (const void *)queries.data(row_index);
            std::vector<std::pair<float, label_t>> top_k = this->_index->search(
                /* query = */ query, /* K = */ K, /* ef_search = */ ef_search,
                /* num_initializations = */ num_initializations);

            for (uint32_t result_id = 0; result_id < K; result_id++) {
              distances[(row_index * K) + result_id] = top_k[result_id].first;
              results[(row_index * K) + result_id] = top_k[result_id].second;
            }
          });
    }

    // Allows to transfer ownership to Python
    py::capsule free_results_when_done(
        results, [](void *ptr) { delete (label_t *)ptr; });
    py::capsule free_distances_when_done(
        distances, [](void *ptr) { delete (float *)ptr; });

    py::array_t<label_t> labels =
        py::array_t<label_t>({num_queries, (size_t)K}, // shape of the array
                             {K * sizeof(label_t), sizeof(label_t)}, // strides
                             results,               // data pointer
                             free_results_when_done // capsule
        );

    py::array_t<float> dists = py::array_t<float>(
        {num_queries, (size_t)K}, {K * sizeof(float), sizeof(float)}, distances,
        free_distances_when_done);

    return {dists, labels};
  }
};

using L2FlatNavIndex = PyIndex<SquaredL2Distance, int>;
using InnerProductFlatNavIndex = PyIndex<InnerProductDistance, int>;

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

static const char *LOAD_DOCSTRING = R"pbdoc(
Load a FlatNav index from a given file location.
Args:
    filename (str): The file location to load the index from.
Returns:
    Union[L2Inde, IPIndex]: The loaded index.
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
    Union[L2Index, IPIndex]: The constructed index.
)pbdoc";

template <typename IndexType>
void bindIndexMethods(
    py::class_<IndexType, std::shared_ptr<IndexType>> &index_class) {
  index_class
      .def(
          "save",
          [](IndexType &index_type, const std::string &filename) {
            auto index = index_type.getIndex();
            index->saveIndex(/* filename = */ filename);
          },
          py::arg("filename"), SAVE_DOCSTRING)
      .def_static("load", &IndexType::loadIndex, py::arg("filename"),
                  LOAD_DOCSTRING)
      .def("add", &IndexType::add, py::arg("data"), py::arg("ef_construction"),
           py::arg("num_initializations") = 100, py::arg("labels") = py::none(),
           ADD_DOCSTRING)
      .def("allocate_nodes", &IndexType::allocateNodes, py::arg("data"),
           ALLOCATE_NODES_DOCSTRING)
      .def("search_single", &IndexType::searchSingle, py::arg("query"),
           py::arg("K"), py::arg("ef_search"),
           py::arg("num_initializations") = 100, SEARCH_SINGLE_DOCSTRING)
      .def("get_query_distance_computations",
           &IndexType::getQueryDistanceComputations,
           GET_QUERY_DISTANCE_COMPUTATIONS_DOCSTRING)
      .def("search", &IndexType::search, py::arg("queries"), py::arg("K"),
           py::arg("ef_search"), py::arg("num_initializations") = 100,
           SEARCH_DOCSTRING)
      .def(
          "get_graph_outdegree_table",
          [](IndexType &index_type) -> std::vector<std::vector<uint32_t>> {
            auto index = index_type.getIndex();
            return index->getGraphOutdegreeTable();
          },
          GET_GRAPH_OUTDEGREE_TABLE_DOCSTRING)
      .def(
          "build_graph_links",
          [](IndexType &index_type, const std::string &mtx_filename) {
            auto index = index_type.getIndex();
            index->buildGraphLinks(/* mtx_filename = */ mtx_filename);
          },
          py::arg("mtx_filename"), BUILD_GRAPH_LINKS_DOCSTRING)
      .def(
          "reorder",
          [](IndexType &index_type,
             const std::vector<std::string> &strategies) {
            auto index = index_type.getIndex();
            // validate the given strategies
            for (auto &strategy : strategies) {
              auto alg = strategy;
              std::transform(alg.begin(), alg.end(), alg.begin(),
                             [](unsigned char c) { return std::tolower(c); });
              if (alg != "gorder" && alg != "rcm") {
                throw std::invalid_argument(
                    "`" + strategy +
                    "` is not a supported graph re-ordering strategy.");
              }
            }
            index->doGraphReordering(strategies);
          },
          py::arg("strategies"), REORDER_DOCSTRING)
      .def(
          "set_num_threads",
          [](IndexType &index_type, uint32_t num_threads) {
            auto *index = index_type.getIndex();
            index->setNumThreads(num_threads);
          },
          py::arg("num_threads"), SET_NUM_THREADS_DOCSTRING)
      .def_property_readonly(
          "num_threads",
          [](IndexType &index_type) {
            auto *index = index_type.getIndex();
            return index->getNumThreads();
          },
          NUM_THREADS_DOCSTRING)
      .def_property_readonly(
          "max_edges_per_node",
          [](IndexType &index_type) {
            return index_type.getIndex()->maxEdgesPerNode();
          },
          MAX_EDGES_PER_NODE_DOCSTRING);
}

template <typename... Args>
py::object createIndex(const std::string &distance_type, int dim,
                       Args &&...args) {
  auto dist_type = distance_type;
  std::transform(dist_type.begin(), dist_type.end(), dist_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (dist_type == "l2") {
    auto distance = std::make_unique<SquaredL2Distance>(/* dim = */ dim);
    return py::cast(std::make_shared<L2FlatNavIndex>(
        std::move(distance), std::forward<Args>(args)...));
  } else if (dist_type == "angular") {
    auto distance = std::make_unique<InnerProductDistance>(/* dim = */ dim);
    return py::cast(std::make_shared<InnerProductFlatNavIndex>(
        std::move(distance), std::forward<Args>(args)...));
  }
  throw std::invalid_argument("Invalid distance type: `" + dist_type +
                              "` during index construction. Valid options "
                              "include `l2` and `angular`.");
}

void defineIndexSubmodule(py::module_ &index_submodule) {
  index_submodule.def(
      "index_factory",
      [](const std::string &distance_type, int dim, int dataset_size,
         int max_edges_per_node, bool verbose = false,
         bool collect_stats = false) {
        return createIndex(distance_type, dim, dataset_size, max_edges_per_node,
                           verbose, collect_stats);
      },
      py::arg("distance_type"), py::arg("dim"), py::arg("dataset_size"),
      py::arg("max_edges_per_node"), py::arg("verbose") = false,
      py::arg("collect_stats") = false, CONSTRUCTOR_DOCSTRING);

  py::class_<L2FlatNavIndex, std::shared_ptr<L2FlatNavIndex>> l2_index_class(
      index_submodule, "L2Index");
  bindIndexMethods(l2_index_class);

  py::class_<InnerProductFlatNavIndex,
             std::shared_ptr<InnerProductFlatNavIndex>>
      ip_index_class(index_submodule, "IPIndex");
  bindIndexMethods(ip_index_class);
}

PYBIND11_MODULE(flatnav, module) {
  auto index_submodule = module.def_submodule("index");

  defineIndexSubmodule(index_submodule);
}