#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/index/PrimitiveTypes.h>
#include <flatnav/index/Pruning.h>
#include <flatnav/util/Datatype.h>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <optional>
#include <cereal/types/optional.hpp>
#include <memory>

namespace flatnav {

struct IndexBuildParameters {
  using node_id_t = flatnav::index_node_id_t;

  size_t dim;
  size_t M;
  size_t dataset_size;
  size_t node_size_bytes;
  size_t data_size_bytes;
  DataType data_type = DataType::float32;
  size_t ef_construction;
  flatnav::PruningHeuristic pruning_heuristic;
  std::optional<float> pruning_heuristic_parameter;

  friend class cereal::access;
  IndexBuildParameters() = default;

  IndexBuildParameters(size_t dim, size_t M, size_t dataset_size, DataType data_type,
                       size_t ef_construction,
                       PruningHeuristic heuristic = PruningHeuristic::ARYA_MOUNT,
                       std::optional<float> parameter = std::nullopt)
      : dim(dim),
        M(M),
        dataset_size(dataset_size),
        data_type(data_type),
        ef_construction(ef_construction),
        pruning_heuristic(heuristic),
        pruning_heuristic_parameter(parameter) {
    // TODO: This is not great because we are hardcoding the label_t to be int.
    // Figure out a better way to handle this.
    data_size_bytes = dim * flatnav::util::size(data_type);
    node_size_bytes = data_size_bytes + (sizeof(node_id_t) * M) + sizeof(int);
  }

  template <class Archive>
  void serialize(Archive& archive) {
    archive(dim, M, dataset_size, data_type, ef_construction, pruning_heuristic,
           pruning_heuristic_parameter);
  }

  static std::shared_ptr<IndexBuildParameters> load(const std::string& filename) {
    std::ifstream stream(filename, std::ios::binary);
    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    auto params = std::make_shared<IndexBuildParameters>();
    cereal::BinaryInputArchive archive(stream);
    archive(*params);
    return params;
  }

  void save(const std::string& filename) {
    std::ofstream stream(filename, std::ios::binary);
    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    cereal::BinaryOutputArchive archive(stream);
    archive(*this);
  }
};

template <typename label_t>
class FlatMemoryAllocator {
  using node_id_t = flatnav::index_node_id_t;
  IndexBuildParameters* _params;
  char* _index_memory;

  FlatMemoryAllocator(const FlatMemoryAllocator&) = delete;
  FlatMemoryAllocator& operator=(const FlatMemoryAllocator&) = delete;

 public:
  FlatMemoryAllocator(IndexBuildParameters* params) : _params(params) {
    uint64_t index_size = static_cast<uint64_t>(_params->node_size_bytes) *
                          static_cast<uint64_t>(_params->dataset_size);
    _index_memory = new char[index_size];
  }

  ~FlatMemoryAllocator() { delete[] _index_memory; }

  char* getIndexMemoryBlock() { return _index_memory; }

  char* getNodeData(node_id_t node_id) const {
    return _index_memory + node_id * _params->node_size_bytes;
  }

  node_id_t* getNodeLinks(node_id_t node_id) const {
    size_t offset = node_id * _params->node_size_bytes + _params->data_size_bytes;
    return reinterpret_cast<node_id_t*>(_index_memory + offset);
  }

  label_t* getNodeLabel(node_id_t node_id) const {
    size_t offset = node_id * _params->node_size_bytes + _params->data_size_bytes +
                    _params->M * sizeof(node_id_t);
    return reinterpret_cast<label_t*>(_index_memory + offset);
  }

  inline uint64_t getTotalIndexMemory() const {
    return static_cast<uint64_t>(_params->node_size_bytes) *
           static_cast<uint64_t>(_params->dataset_size);
  }

  const IndexBuildParameters& getParams() const { return *_params; }
};

}  // namespace flatnav