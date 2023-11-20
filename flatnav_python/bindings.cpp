#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <utility>
#include <vector>

#include <flatnav/DistanceInterface.h>
#include <flatnav/Index.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>

using flatnav::DistanceInterface;
using flatnav::Index;
using flatnav::InnerProductDistance;
using flatnav::SquaredL2Distance;

namespace py = pybind11;

template <typename dist_t, typename label_t> class PythonIndex {
private:
  int _dim, label_id;
  Index<dist_t, label_t> *_index;

public:
  PythonIndex(std::shared_ptr<DistanceInterface<dist_t>> distance, int dim,
              int dataset_size, int max_edges_per_node)
      : _dim(dim), label_id(0),
        _index(new Index<dist_t, label_t>(
            /* dist = */ std::move(distance),
            /* dataset_size = */ dataset_size,
            /* max_edges_per_node = */ max_edges_per_node)) {}

  Index<dist_t, label_t> *getIndex() const { return _index; }

  ~PythonIndex() { delete _index; }

  void
  add(const py::array_t<float, py::array::c_style | py::array::forcecast> &data,
      int ef_construction, py::object labels = py::none()) {
    // py::array_t<float, py::array::c_style | py::array::forcecast> means that
    // the functions expects either a Numpy array of floats or a castable type
    // to that type. If the given type can't be casted, pybind11 will throw an
    // error.

    auto num_vectors = data.shape(0);
    auto data_dim = data.shape(1);
    if (data.ndim() != 2 || data_dim != _dim) {
      throw std::invalid_argument("Data has incorrect dimensions.");
    }

    if (labels.is_none()) {
      for (size_t vec_index = 0; vec_index < num_vectors; vec_index++) {
        this->_index->add(/* data = */ (void *)data.data(vec_index),
                          /* label = */ label_id,
                          /* ef_construction = */ ef_construction);
        label_id++;
      }
      return;
    }

    // Use the provided labels now
    py::array_t<label_t, py::array::c_style | py::array::forcecast> node_labels(
        labels);
    if (node_labels.ndim() != 1 || node_labels.shape(0) != num_vectors) {
      throw std::invalid_argument("Labels have incorrect dimensions.");
    }

    for (size_t vec_index = 0; vec_index < num_vectors; vec_index++) {
      label_t label_id = *labels.data(vec_index);
      this->_index->add(/* data = */ (void *)data.data(vec_index),
                        /* label = */ label_id,
                        /* ef_construction = */ ef_construction);
    }
  }

  py::array_t<label_t>
  search(const py::array_t<float, py::array::c_style | py::array::forcecast>
             queries,
         int K, int ef_search) {
    auto num_queries = queries.shape(0);
    auto queries_dim = queries.shape(1);

    if (queries.ndim() != 2 || queries_dim != _dim) {
      throw std::invalid_argument("Queries have incorrect dimensions.");
    }

    label_t *results = new label_t[num_queries * K];
    // float *distances = new float[num_queries * K];

    for (size_t query_index; query_index < num_queries; query_index++) {
      std::vector<std::pair<float, label_t>> top_k = this->_index->search(
          /* query = */ (const void *)queries.data(query_index), /* K = */ K,
          /* ef_search = */ ef_search);

      for (size_t i = 0; i < top_k.size(); i++) {
        // distances[query_index * K + i] = top_k[results].first;
        results[query_index * K + i] = top_k[results].second;
      }
    }

    py::capsule free_when_done(results, [](void *ptr) { delete ptr; });

    return py::array_t<label_t>({num_queries, (size_t)K},
                                {K * sizeof(label_t), sizeof(label_t)}, results,
                                free_when_done);
  }
};

using L2FlatNavIndex = PythonIndex<SquaredL2Distance, int>;
using InnerProductFlatNavIndex = PythonIndex<InnerProductDistance, int>;

template <typename IndexType>
void bindIndexMethods(py::class_<IndexType> &index_class){
    index_class
        .def("add", &IndexType::add, py::arg("data"),
             py::arg("ef_construction"), py::arg("labels") = py::none(),
             "Add vectors(data) to the index with the given `ef_construction` "
             "parameter and optional labels. `ef_construction` determines how "
             "many "
             "vertices are visited while inserting every vector in the "
             "underlying graph structure.")
        .def("search", &IndexType::search, py::arg("queries"), py::arg("K"),
             py::arg("ef_search"),
             "Return top `K` closest data points for every query in the "
             "provided `queries`. The `ef_search` parameter determines how "
             "many neighbors are visited while finding the closest neighbors "
             "for every query.");

}

py::object createIndex(const std::string &distance_type, int dim,
                       int dataset_size, int max_edges_per_node) {
  auto dist_type = distance_type;
  std::transform(dist_type.begin(), dist_type.end(), dist_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (dist_type == "l2") {
    auto distance = std::make_shared<SquaredL2Distance>(/* dim = */ dim);
    return py::cast(new L2FlatNavIndex(std::move(distance), dim, dataset_size,
                                       max_edges_per_node));
  } else if (dist_type == "angular") {
    auto distance = std::make_shared<InnerProductDistance>(/* dim = */ dim);
    return py::cast(new InnerProductFlatNavIndex(
        std::move(distance), dim, dataset_size, max_edges_per_node));
  }
  throw std::invalid_argument("Invalid distance type: `" + dist_type +
                              "` during index construction. Valid options "
                              "include `l2` and `angular`.");
}

void defineIndexSubmodule(py::module_ &index_submodule) {
  index_submodule.def("index_factory", &createIndex, py::arg("distance_type"),
                      py::arg("dim"), py::arg("dataset_size"),
                      py::arg("max_edge_per_node"),
                      "Creates a FlatNav index given the corresponding "
                      "parameters. The `distance_type` argument determines the "
                      "kind of index created (either L2Index or IPIndex)");

  py::class_<L2FlatNavIndex> l2_index_class(index_submodule, "L2Index");
  bindIndexMethods(l2_index_class);

  py::class_<InnerProductFlatNavIndex> ip_index_class(index_submodule,
                                                      "IPIndex");
  bindIndexMethods(ip_index_class);
}

PYBIND11_MODULE(flatnav, module) {
  auto index_submodule = module.def_submodule("index");

  defineIndexSubmodule(index_submodule);
}