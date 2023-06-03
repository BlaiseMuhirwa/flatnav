#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../flatnav/Index.h"
#include "../flatnav/distances/InnerProductDistance.h"
#include "../flatnav/distances/SquaredL2Distance.h"
#include "../flatnav/distances/SquaredL2DistanceSpecializations.h"

using namespace flatnav;
namespace py = pybind11;

template <typename dist_t, typename label_t> class PyIndex {
private:
  Index<dist_t, label_t> *_index;
  DistanceInterface<dist_t> _distance;

  size_t _dim;
  int _added;

  void setIndexMetric(std::string &metric) {
    std::transform(metric.begin(), metric.end(), metric.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (metric == "l2") {
      _distance = std::move(SquaredL2Distance(/* dim = */ _dim));
    } else if (metric == "angular") {
      _distance = std::move(InnerProductDistance(/* dim = */ _dim));
    }
    throw std::invalid_argument("Invalid metric `" + metric +
                                "` used during index construction.");
  }

public:
  PyIndex(std::string metric_type, size_t dim, int N, int M)
      : _dim(dim), _added(0) {
    setIndexMetric(metric_type);
    _index = new Index<dist_t, label_t>(
        /* dist = */ _distance, /* num_data = */ N, /* max_edges = */ M);
  }

  PyIndex(std::string filename) {
    _index = new Index<dist_t, label_t>(/* in = */ filename);
  }

  void add(py::array_t<float, py::array::c_style | py::array::forcecast> data,
           int ef_construction, py::object labels_obj = py::none()) {

    if (data.n_dim() != 2 || data.shape(1) != _dim) {
      throw std::invalid_argument("Data has incorrect _dimensions");
    }

    if (labels_obj.is_none()) {
      for (size_t n = 0; n < data.shape(0); n++) {
        this->index->add((void *)data.data(n), _added, ef_construction);
        _added++;
      }
    } else {
      py::array_t<label_t, py::array::c_style | py::array::forcecast> labels(
          labels_obj);
      if (labels.n_dim() != 1 || labels.shape(0) != data.shape(0)) {
        throw std::invalid_argument("Labels have incorrect _dimensions");
      }

      for (size_t n = 0; n < data.shape(0); n++) {
        label_t l = *labels.data(n);
        this->index->add((void *)data.data(n), l, ef_construction);
        _added++;
      }
    }
  }

  py::array_t<label_t>
  search(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
         int K, int ef_search) {
    if (queries.n_dim() != 2 || queries.shape(1) != _dim) {
      throw std::invalid_argument("Queries have incorrect _dimensions");
    }
    size_t num_queries = queries.shape(0);

    label_t *results = new label_t[num_queries * K];

    for (size_t q = 0; q < num_queries; q++) {
      std::vector<std::pair<dist_t, label_t>> topK =
          this->index->search(queries.data(q), K, ef_search);
      for (size_t i = 0; i < topK.size(); i++) {
        results[q * K + i] = topK[i].second;
      }
    }

    py::capsule free_when_done(results, [](void *ptr) { delete ptr; });

    return py::array_t<label_t>({num_queries, (size_t)K},
                                {K * sizeof(label_t), sizeof(label_t)}, results,
                                free_when_done);
  }

  void reorder(std::string alg) {
    std::transform(alg.begin(), alg.end(), std::tolower);

    if (alg == "gorder") {
      this->index->reorder_gorder();
    } else if (alg == "rcm") {
      this->index->reorder_rcm();
    } else {
      throw std::invalid_argument(
          "'" + alg + "' is not a supported graph re-ordering algorithm.");
    }
  }

  void save(std::string filename) { this->index->save(filename); }

  ~PyIndex() {
    delete index;
    delete space;
  }
};

template <typename label_t>
double ComputeRecall(py::array_t<label_t> results,
                     py::array_t<label_t> gtruths) {
  double avg_recall = 0.0;
  for (size_t q = 0; q < results.shape(0); q++) {
    double recall = 0.0;
    const label_t *result = results.data(q);
    const label_t *topk = gtruths.data(q);
    for (size_t i = 0; i < results.shape(1); i++) {
      for (size_t j = 0; j < results.shape(1); j++) {
        if (result[i] == topk[j]) {
          recall += 1.0;
          break;
        }
      }
    }
    avg_recall += recall;
  }

  return avg_recall /= (results.shape(0) * results.shape(1));
}

using L2FloatPyIndex = PyIndex<SquaredL2Distance, unsigned int>;

PYBIND11_MODULE(flatnav, m) {
  py::class_<L2FloatPyIndex>(m, "Index")
      .def(py::init<std::string, size_t, int, int>(), py::arg("metric"),
           py::arg("_dim"), py::arg("N"), py::arg("M"))
      .def(py::init<std::string>(), py::arg("save_loc"))
      .def("add", &L2FloatPyIndex::add, py::arg("data"),
           py::arg("ef_construction"), py::arg("labels") = py::none())
      .def("search", &L2FloatPyIndex::search, py::arg("queries"), py::arg("K"),
           py::arg("ef_search"))
      .def("reorder", &L2FloatPyIndex::reorder, py::arg("alg"))
      .def("save", &L2FloatPyIndex::save, py::arg("filename"));

  // m.def("ComputeRecall", &ComputeRecall<int>, py::arg("results"),
  //       py::arg("gtruths"));
}
