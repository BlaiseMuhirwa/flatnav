#include <stdexcept>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../flatnav/Index.h"
#include "../flatnav/distances/SquaredL2Distance.h"
#include "../flatnav/distances/SquaredL2DistanceSpecializations.h"

using namespace flatnav;
namespace py = pybind11;

using L2FloatIndex = Index<SquaredL2Distance, unsigned int>;



template<typename dist_t, typename label_t>
class PyIndex {
  private:
    Index<dist_t, label_t>* index;
    SpaceInterface<dist_t>* space;
    size_t dim;
    int added;

    void getSpaceFromType(std::string& spaceType) {
      if (spaceType == "L2") {
        space = new L2Space(dim);
      } else if (spaceType == "Angular") {
        space = new InnerProductSpace(dim);
      } else {
        throw std::invalid_argument("Invalid Space '" + spaceType + "' used to construct Index");
      }
    }
  
public:
  PyIndex(std::string spaceType, size_t _dim, int _N, int _M): dim(_dim), added(0) {
    getSpaceFromType(spaceType);
    index = new Index<dist_t, label_t>(space, _N, _M);
	}

  PyIndex(std::string spaceType, size_t _dim, std::string filename): dim(_dim) {
    getSpaceFromType(spaceType);
    index = new Index<dist_t, label_t>(space, filename);
  }

  void Add(
    py::array_t<float, py::array::c_style | py::array::forcecast> data, 
    int ef_construction, 
    py::object labels_obj = py::none()) {

    if (data.ndim() != 2 || data.shape(1) != dim) {
      throw std::invalid_argument("Data has incorrect dimensions");
    }

    if (labels_obj.is_none())  {
      for (size_t n = 0; n < data.shape(0); n++) {
        this->index->add((void*)data.data(n), added, ef_construction);  
        added++;
      }    
    } else {  
      py::array_t<label_t, py::array::c_style | py::array::forcecast> labels(labels_obj);
      if (labels.ndim() != 1 || labels.shape(0) != data.shape(0)) {
        throw std::invalid_argument("Labels have incorrect dimensions");
      }

      for (size_t n = 0; n < data.shape(0); n++) {
        label_t l = *labels.data(n);
        this->index->add((void*)data.data(n), l, ef_construction);  
        added++;
      }  
    }
  }

  py::array_t<label_t> Search(py::array_t<float, py::array::c_style | py::array::forcecast> queries, int K, int ef_search) {
    if (queries.ndim() != 2 || queries.shape(1) != dim) {
      throw std::invalid_argument("Queries have incorrect dimensions");
    }
    size_t num_queries = queries.shape(0);

    label_t* results = new label_t[num_queries * K];

    for (size_t q = 0; q < num_queries; q++) {
      std::vector<std::pair<dist_t, label_t>> topK = this->index->search(queries.data(q), K, ef_search);
      for (size_t i = 0; i < topK.size(); i++) {
        results[q * K + i] = topK[i].second;
      }
    }

    py::capsule free_when_done(results, [](void* ptr){ delete ptr;});

    return py::array_t<label_t>(
      {num_queries,(size_t) K},
      {K * sizeof(label_t), sizeof(label_t)},
      results,
      free_when_done
    );
  }

  void Reorder(std::string alg) {
      if (alg =="gorder") {
        this->index->reorder(Index<dist_t, label_t>::GraphOrder::GORDER);
      } else if (alg == "in_deg") {
        this->index->reorder(Index<dist_t, label_t>::GraphOrder::IN_DEG);
      } else if (alg == "out_deg") {
        this->index->reorder(Index<dist_t, label_t>::GraphOrder::OUT_DEG);
      } else if (alg == "rcm") {
        this->index->reorder(Index<dist_t, label_t>::GraphOrder::RCM);
      } else if (alg == "hub_sort") {
        this->index->reorder(Index<dist_t, label_t>::GraphOrder::HUB_SORT);
      } else if (alg == "hub_cluster") {
        this->index->reorder(Index<dist_t, label_t>::GraphOrder::HUB_CLUSTER);
      } else if (alg == "DBG") {
        this->index->reorder(Index<dist_t, label_t>::GraphOrder::DBG);
      } else {
        throw std::invalid_argument("'" + alg + "' is not a supported graph reordering algorithm");
      }
  }

  void Save(std::string filename) {
    this->index->save(filename);
  }

  ~PyIndex() {
    delete index;
    delete space;
  }
};

template<typename label_t>
double ComputeRecall(py::array_t<label_t> results, py::array_t<label_t> gtruths) {
  double avg_recall = 0.0;
  for (size_t q = 0; q < results.shape(0); q++) {
    double recall = 0.0;
    const label_t* result = results.data(q);
    const label_t* topk = gtruths.data(q);
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

using PyIndexWithTypes = PyIndex<float, int>;

PYBIND11_MODULE(flatnav, m) {
  py::class_<PyIndexWithTypes>(m, "Index")
      .def(py::init<std::string, size_t, int, int>(), py::arg("space"), py::arg("dim"), py::arg("N"), py::arg("M"))
      .def(py::init<std::string, size_t, std::string>(), py::arg("space"), py::arg("dim"), py::arg("save_loc"))
      .def("Add", &PyIndexWithTypes::Add, py::arg("data"), py::arg("ef_construction"), py::arg("labels")=py::none())
      .def("Search", &PyIndexWithTypes::Search, py::arg("queries"), py::arg("K"), py::arg("ef_search"))
      .def("Reorder", &PyIndexWithTypes::Reorder, py::arg("alg"))
      .def("Save", &PyIndexWithTypes::Save, py::arg("filename"));

  m.def("ComputeRecall", &ComputeRecall<int>, py::arg("results"), py::arg("gtruths"));
}
