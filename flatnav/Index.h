#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <condition_variable>
#include <cstring>
#include <flatnav/DistanceInterface.h>
#include <flatnav/util/ParallelConstructs.h>
#include <flatnav/util/PreprocessorUtils.h>
#include <flatnav/util/Reordering.h>
#include <flatnav/util/SIMDDistanceSpecializations.h>
#include <flatnav/util/VisitedSetPool.h>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace flatnav {
using IndexParameterValue = std::variant<uint32_t, float, std::string>;
using ParameterMap = std::unordered_map<std::string, IndexParameterValue>;

//clang-format off
/**
 * @brief A builder class for constructing an index object with various
 * configuration parameters.
 *
 * The IndexParameterConfig class follows the builder pattern to allow for flexible
 * configuration of an index object before its instantiation. It encapsulates
 * all the necessary parameters required to create an index, such as the
 * distance metric, dataset size, and various optimization parameters. The class
 * is designed to be used in a chained manner, where methods are called in
 * sequence to set different configuration options. This approach makes the code
 * more readable and maintainable.
 *
 * @tparam dist_t The data type of the distance metric (e.g., float, double).
 *
 * Usage:
 * 1. Instantiate an IndexParameterConfig object with the required distance metric and
 * dataset size.
 * 2. Optionally configure the index with additional parameters such as max
 * edges per node, ef construction, number of initializations, number of
 * threads, and graph reordering methods.
 * 3. Call the `build` method to construct the configured index object.
 *
 * Example:
 * ```cpp
 * auto distance = std::make_shared<SquaredL2Distance>(dim=128);
 * std::shared_ptr<IndexParameterConfig> builder =
 *     IndexParameterConfig::create(std::move(distance), N)
 *         ->withEfConstruction(ef_construction)
 *         ->withMaxEdgesPerNode(M)
 *         ->withGraphReordering({"gorder", "rcm"})
 *         ->withNumThreads(std::thread::hardware_concurrency());
 * ```
 *
 * Notes:
 * - The builder pattern provides a clear and flexible way to construct objects
 * when the object creation process involves multiple steps or when many
 * parameters are required.
 * - This implementation allows for easy extension and addition of new
 * parameters without affecting existing code.
 * - The methods `withMaxEdgesPerNode`, `withEfConstruction`, `withNumThreads`,
 * and `withGraphReordering` return a shared pointer to the builder itself,
 * allowing for the chaining of method calls.
 * - The `build` method finalizes the configuration and returns a shared pointer
 * to the constructed index object.
 */

struct IndexParameterConfig : public std::enable_shared_from_this<IndexParameterConfig> {

  uint32_t max_edges_per_node;
  uint32_t ef_construction;
  uint32_t max_node_count;
  uint32_t num_initializations;
  uint32_t num_threads;
  std::optional<std::vector<std::string>> reordering_methods;
  std::optional<std::string> index_name;

  IndexParameterConfig() = default;
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(max_edges_per_node, ef_construction, max_node_count,
            num_initializations, reordering_methods, index_name);
  }

  void saveIndexParameterConfig(const std::string &filename) {
    std::ofstream stream(filename, std::ios::binary);
    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }
    cereal::BinaryOutputArchive archive(stream);
    archive(*this);
  }

  static std::shared_ptr<IndexParameterConfig>
  loadIndexParameterConfig(const std::string &filename) {
    std::ifstream stream(filename, std::ios::binary);
    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }
    cereal::BinaryInputArchive archive(stream);
    std::shared_ptr<IndexParameterConfig> builder(new IndexParameterConfig());
    archive(*builder);
    return builder;
  }

  static std::shared_ptr<IndexParameterConfig> create(uint32_t dataset_size) {
    return std::make_shared<IndexParameterConfig>(dataset_size);
  }

  explicit IndexParameterConfig(uint32_t dataset_size)
      : max_node_count(dataset_size), num_initializations(100), num_threads(1) {
  }

  std::shared_ptr<IndexParameterConfig> withMaxEdgesPerNode(uint32_t max_edges) {
    max_edges_per_node = max_edges;
    return this->shared_from_this();
  }

  std::shared_ptr<IndexParameterConfig> withEfConstruction(uint32_t value) {
    ef_construction = value;
    return this->shared_from_this();
  }

  std::shared_ptr<IndexParameterConfig> withIndexParams(const ParameterMap &params) {
    for (const auto &[key, value] : params) {
      if (key == "max_edges_per_node") {
        max_edges_per_node = std::get<uint32_t>(value);
      } else if (key == "num_threads") {
        num_threads = std::get<uint32_t>(value);
      } else if (key == "ef_construction") {
        ef_construction = std::get<uint32_t>(value);
      } else if (key == "num_initializations") {
        num_initializations = std::get<uint32_t>(value);
      } else if (key == "index_name") {
        index_name = std::get<std::string>(value);
      } else {
        throw std::invalid_argument("Invalid parameter: " + key);
      }
    }

    return this->shared_from_this();
  }

  std::shared_ptr<IndexParameterConfig>
  withGraphReordering(const std::vector<std::string> &values) {
    for (const auto &method : values) {
      if (method != "gorder" && method != "rcm") {
        throw std::invalid_argument("Invalid reordering method: " + method);
      }
    }
    reordering_methods = std::move(values);
    return this->shared_from_this();
  }

  std::shared_ptr<IndexParameterConfig> withNumThreads(int threads) {
    if (threads == 0 || threads > std::thread::hardware_concurrency()) {
      throw std::invalid_argument(
          "Number of threads must be greater than 0 and less than or equal to "
          "the number of hardware threads.");
    }
    num_threads = threads;
    return this->shared_from_this();
  }
};

// dist_t: A distance function implementing DistanceInterface.
// label_t: A fixed-width data type for the label (meta-data) of each point.
template <typename dist_t, typename label_t> class Index {
  typedef std::pair<float, label_t> dist_label_t;
  // internal node numbering scheme. We might need to change this to uint64_t
  typedef uint32_t node_id_t;
  typedef std::pair<float, node_id_t> dist_node_t;

  // NOTE: by default this is a max-heap. We could make this a min-heap
  // by using std::greater, but we want to use the queue as both a max-heap and
  // min-heap depending on the context.
  typedef std::priority_queue<dist_node_t, std::vector<dist_node_t>>
      PriorityQueue;

  std::shared_ptr<DistanceInterface<dist_t>> _distance;
  std::shared_ptr<IndexParameterConfig> _parameter_config;
  // Large (several GB), pre-allocated block of memory.
  char *_index_memory;

  // size of one data point (does not support variable-size data, strings)
  size_t _data_size_bytes;
  // Node consists of: ([data] [M links] [data label]). This layout was chosen
  // after benchmarking - it's slightly more cache-efficient than others.
  size_t _node_size_bytes;
  size_t _cur_num_nodes;
  std::mutex _index_data_guard;

  // Remembers which nodes we've visited, to avoid re-computing distances.
  VisitedSetPool *_visited_set_pool;
  std::vector<std::mutex> _node_links_mutexes;

  template <typename Archive> void serialize(Archive &archive) {
    archive(*_distance, _data_size_bytes, _node_size_bytes, _cur_num_nodes);

    // Serialize the allocated memory for the index & query.
    archive(cereal::binary_data(
        _index_memory, _node_size_bytes * _parameter_config->max_node_count));
  }

public:
  /**
   * @brief Construct a new Index object using an IndexParameterConfig.
   * @param parameter_config An IndexParameterConfig object containing the configuration
   * parameters for the index.
   * NOTE: This constructor does not use std::move() because we want to keep the
   * parameter_config object alive for the lifetime of the Index object.
   * The builder object could be modified after the Index object is constructed.
   *
   */
  Index(std::shared_ptr<DistanceInterface<dist_t>> distance,
        std::shared_ptr<IndexParameterConfig> parameter_config)
      : _distance(std::move(distance)), _parameter_config(parameter_config),
        _cur_num_nodes(0),
        _visited_set_pool(new VisitedSetPool(
            /* initial_pool_size = */ 1,
            /* num_elements = */ _parameter_config->max_node_count)),
        _node_links_mutexes(_parameter_config->max_node_count) {
    _data_size_bytes = _distance->dataSize();
    _node_size_bytes =
        _data_size_bytes +
        (sizeof(node_id_t) * _parameter_config->max_edges_per_node) +
        sizeof(label_t);
    size_t index_memory_size =
        _node_size_bytes * _parameter_config->max_node_count;
    _index_memory = new char[index_memory_size];
  }

  ~Index() {
    delete[] _index_memory;
    delete _visited_set_pool;
  }

  void
  buildGraphLinks(const std::vector<std::vector<uint32_t>> &outdegree_table) {
    auto M = _parameter_config->max_edges_per_node;
    for (node_id_t node = 0; node < outdegree_table.size(); node++) {
      node_id_t *links = getNodeLinks(node);
      for (int i = 0; i < M; i++) {
        if (i >= outdegree_table[node].size()) {
          links[i] = node;
        } else {
          links[i] = outdegree_table[node][i];
        }
      }
    }
  }

  std::vector<std::vector<uint32_t>> getGraphOutdegreeTable() {
    std::vector<std::vector<uint32_t>> outdegree_table(_cur_num_nodes);
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      node_id_t *links = getNodeLinks(node);
      for (int i = 0; i < _parameter_config->max_edges_per_node; i++) {
        if (links[i] != node) {
          outdegree_table[node].push_back(links[i]);
        }
      }
    }
    return outdegree_table;
  }

  /**
   * @brief Store the new node in the global data structure. In a
   * multi-threaded setting, the index data guard should be held by the caller
   * with an exclusive lock.
   *
   * @param data The vector to add.
   * @param label The label (meta-data) of the vector.
   * @param new_node_id The id of the new node.
   */
  void allocateNode(void *data, label_t &label, node_id_t &new_node_id) {
    new_node_id = _cur_num_nodes;
    _distance->transformData(
        /* destination = */ getNodeData(new_node_id),
        /* src = */ data);
    *(getNodeLabel(new_node_id)) = label;

    node_id_t *links = getNodeLinks(new_node_id);
    // Initialize all edges to self
    std::fill_n(links, _parameter_config->max_edges_per_node, new_node_id);
    _cur_num_nodes++;
  }

  /**
   * @brief Adds vectors to the index in batches.
   *
   * This method is responsible for adding vectors in batches, represented by
   * `data`, to the underlying graph. Each vector is associated with a label
   * provided in the `labels` vector. The method efficiently handles concurrent
   * additions by dividing the workload among multiple threads, defined by
   * `_parameter_config->num_threads`.
   *
   * The method ensures thread safety by employing locking mechanisms at the
   * node level in the underlying `connectNeighbors` and `beamSearch` methods.
   * This allows multiple threads to safely add vectors to the index without
   * causing data races or inconsistencies in the graph structure.
   *
   * @param data Pointer to the array of vectors to be added.
   * @param labels A vector of labels corresponding to each vector in `data`.
   *
   * @exception std::invalid_argument Thrown if `num_initializations` is less
   * than or equal to 0.
   * @exception std::runtime_error Thrown if the maximum number of nodes in the
   * index is reached.
   */
  void addBatch(void *data, std::vector<label_t> &labels) {
    uint32_t total_num_nodes = labels.size();
    uint32_t data_dimension = _distance->dimension();

    // Don't spawn any threads if we are only using one.
    if (_parameter_config->num_threads == 1) {
      for (uint32_t row_id = 0; row_id < total_num_nodes; row_id++) {
        void *vector = (float *)data + (row_id * data_dimension);
        label_t label = labels[row_id];
        this->add(vector, label);
      }
      return;
    }

    flatnav::executeInParallel(
        /* start_index = */ 0, /* end_index = */ total_num_nodes,
        /* num_threads = */ _parameter_config->num_threads, /* function = */
        [&](uint32_t row_index) {
          void *vector = (float *)data + (row_index * data_dimension);
          label_t label = labels[row_index];
          this->add(vector, label);
        });
  }

  /**
   * @brief Adds a single vector to the index.
   *
   * This method is called internally by `addBatch` for each vector in the
   * batch. The method ensures thread safety by using locking primitives,
   * allowing it to be safely used in a multi-threaded environment.
   *
   * The method first checks if the current number of nodes has reached the
   * maximum capacity. If so, it throws a runtime error. It then locks the index
   * structure to prevent concurrent modifications while allocating a new node.
   * After unlocking, it connects the new node to its neighbors in the graph.
   *
   * @param data Pointer to the vector data being added.
   * @param label Label associated with the vector.
   * @param ef_construction Parameter controlling the size of the dynamic
   * candidate list during the construction of the graph.
   * @param num_initializations Number of initializations for the search
   * algorithm.
   *
   * @exception std::runtime_error Thrown if the maximum number of nodes is
   * reached.
   */
  void add(void *data, label_t &label) {
    if (_cur_num_nodes >= _parameter_config->max_node_count) {
      throw std::runtime_error("Maximum number of nodes reached. Consider "
                               "increasing the `max_node_count` parameter to "
                               "create a larger index.");
    }
    _index_data_guard.lock();
    auto entry_node =
        initializeSearch(data, _parameter_config->num_initializations);
    node_id_t new_node_id;
    allocateNode(data, label, new_node_id);
    _index_data_guard.unlock();

    if (new_node_id == 0) {
      return;
    }

    auto neighbors = beamSearch(
        /* query = */ data, /* entry_node = */ entry_node,
        /* buffer_size = */ _parameter_config->ef_construction);

    selectNeighbors(/* neighbors = */ neighbors);
    connectNeighbors(neighbors, new_node_id);
  }

  /***
   * @brief Search the index for the k nearest neighbors of the query.
   * @param query The query vector.
   * @param K The number of nearest neighbors to return.
   * @param ef_search The search beam width.
   * @param num_initializations The number of random initializations to use.
   */
  std::vector<dist_label_t> search(const void *query, const int K,
                                   int ef_search) {
    int num_initializations = _parameter_config->num_initializations;
    node_id_t entry_node = initializeSearch(query, num_initializations);
    PriorityQueue neighbors =
        beamSearch(/* query = */ query,
                   /* entry_node = */ entry_node,
                   /* buffer_size = */ std::max(ef_search, K));
    auto size = neighbors.size();
    while (neighbors.size() > K) {
      neighbors.pop();
    }
    std::vector<dist_label_t> results;
    results.reserve(size);
    while (neighbors.size() > 0) {
      results.emplace_back(neighbors.top().first,
                           *getNodeLabel(neighbors.top().second));
      neighbors.pop();
    }
    std::sort(results.begin(), results.end(),
              [](const dist_label_t &left, const dist_label_t &right) {
                return left.first < right.first;
              });
    return results;
  }

  void doGraphReordering() {
    if (!_parameter_config->reordering_methods.has_value()) {
      throw std::runtime_error("Reordering methods should be specified in the "
                               "IndexParameterConfig object.");
    }

    for (const auto &method : _parameter_config->reordering_methods.value()) {
      auto outdegree_table = getGraphOutdegreeTable();
      std::vector<node_id_t> P;
      if (method == "gorder") {
        P = std::move(flatnav::gOrder<node_id_t>(outdegree_table, 5));
      } else if (method == "rcm") {
        P = std::move(flatnav::rcmOrder<node_id_t>(outdegree_table));
      } else {
        throw std::invalid_argument("Invalid reordering method: " + method);
      }

      relabel(P);
    }
  }

  static std::unique_ptr<Index<dist_t, label_t>>
  loadIndex(const std::string &filename) {
    std::ifstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    cereal::BinaryInputArchive archive(stream);
    std::unique_ptr<Index<dist_t, label_t>> index(new Index<dist_t, label_t>());

    std::string builder_filename = filename + ".builder";
    index->_parameter_config = IndexParameterConfig::loadIndexParameterConfig(
        /* filename = */ builder_filename);
    std::shared_ptr<DistanceInterface<dist_t>> distance =
        std::make_shared<dist_t>();

    // 1. Deserialize metadata
    archive(*distance, index->_data_size_bytes, index->_node_size_bytes,
            index->_cur_num_nodes);
    index->_distance = std::move(distance);
    index->_visited_set_pool = new VisitedSetPool(
        /* initial_pool_size = */ 1,
        /* num_elements = */ index->_parameter_config->max_node_count);

    index->_node_links_mutexes =
        std::vector<std::mutex>(index->_parameter_config->max_node_count);

    // 2. Allocate memory using deserialized metadata
    index->_index_memory = new char[index->_node_size_bytes *
                                    index->_parameter_config->max_node_count];

    // 3. Deserialize content into allocated memory
    archive(cereal::binary_data(index->_index_memory,
                                index->_node_size_bytes *
                                    index->_parameter_config->max_node_count));

    return index;
  }

  void saveIndex(const std::string &filename) {
    std::ofstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    // Save builder object in filename.builder
    std::string builder_filename = filename + ".builder";
    _parameter_config->saveIndexParameterConfig(builder_filename);

    cereal::BinaryOutputArchive archive(stream);
    archive(*this);
  }

  inline size_t maxEdgesPerNode() const {
    return _parameter_config->max_edges_per_node;
  }

  inline void setVisitedPoolSize(uint32_t pool_size) {
    _visited_set_pool->setPoolSize(pool_size);
  }

  inline size_t dataSizeBytes() const { return _data_size_bytes; }

  inline size_t nodeSizeBytes() const { return _node_size_bytes; }

  inline size_t maxNodeCount() const { return _parameter_config->max_node_count; }

  inline size_t currentNumNodes() const { return _cur_num_nodes; }
  inline size_t dataDimension() const { return _distance->dimension(); }

  inline std::shared_ptr<IndexParameterConfig> getBuilder() const {
    return _parameter_config;
  }

  void getIndexSummary() const {
    std::cout << "\nIndex Parameters\n" << std::flush;
    std::cout << "-----------------------------\n" << std::flush;
    std::cout << "max_edges_per_node (M): "
              << _parameter_config->max_edges_per_node << "\n"
              << std::flush;
    std::cout << "data_size_bytes: " << _data_size_bytes << "\n" << std::flush;
    std::cout << "node_size_bytes: " << _node_size_bytes << "\n" << std::flush;
    std::cout << "max_node_count: " << _parameter_config->max_node_count << "\n"
              << std::flush;
    std::cout << "cur_num_nodes: " << _cur_num_nodes << "\n" << std::flush;

    _distance->getSummary();
  }

private:
  friend class cereal::access;
  // Default constructor for cereal
  Index() = default;

  char *getNodeData(const node_id_t &n) const {
    return _index_memory + (n * _node_size_bytes);
  }

  node_id_t *getNodeLinks(const node_id_t &n) const {
    char *location = _index_memory + (n * _node_size_bytes) + _data_size_bytes;
    return reinterpret_cast<node_id_t *>(location);
  }

  label_t *getNodeLabel(const node_id_t &n) const {
    char *location = _index_memory + (n * _node_size_bytes) + _data_size_bytes +
                     (_parameter_config->max_edges_per_node * sizeof(node_id_t));
    return reinterpret_cast<label_t *>(location);
  }

  inline void swapNodes(node_id_t a, node_id_t b, void *temp_data,
                        node_id_t *temp_links, label_t *temp_label) {
    // stash b in temp
    std::memcpy(temp_data, getNodeData(b), _data_size_bytes);
    std::memcpy(temp_links, getNodeLinks(b),
                _parameter_config->max_edges_per_node * sizeof(node_id_t));
    std::memcpy(temp_label, getNodeLabel(b), sizeof(label_t));

    // place node at a in b
    std::memcpy(getNodeData(b), getNodeData(a), _data_size_bytes);
    std::memcpy(getNodeLinks(b), getNodeLinks(a),
                _parameter_config->max_edges_per_node * sizeof(node_id_t));
    std::memcpy(getNodeLabel(b), getNodeLabel(a), sizeof(label_t));

    // put node b in a
    std::memcpy(getNodeData(a), temp_data, _data_size_bytes);
    std::memcpy(getNodeLinks(a), temp_links,
                _parameter_config->max_edges_per_node * sizeof(node_id_t));
    std::memcpy(getNodeLabel(a), temp_label, sizeof(label_t));
  }

  /**
   * @brief Performs beam search for the nearest neighbors of the query.
   * @TODO: Add `entry_node_dist` argument to this function since we expect to
   * have computed that a priori.
   *
   * @param query               The query vector.
   * @param entry_node          The node to start the search from.
   * @param buffer_size         This is equivalent to `ef_search` in the HNSW
   *
   * @return PriorityQueue
   */
  PriorityQueue beamSearch(const void *query, const node_id_t entry_node,
                           const int buffer_size) {
    PriorityQueue neighbors;
    PriorityQueue candidates;

    auto *visited_set = _visited_set_pool->pollAvailableSet();
    visited_set->clear();

    float dist = _distance->distance(
        /* x = */ query, /* y = */ getNodeData(entry_node),
        /* asymmetric = */ true);

    float max_dist = dist;
    candidates.emplace(-dist, entry_node);
    neighbors.emplace(dist, entry_node);
    visited_set->insert(entry_node);

    while (!candidates.empty()) {
      dist_node_t d_node = candidates.top();

      if ((-d_node.first) > max_dist && neighbors.size() >= buffer_size) {
        break;
      }
      candidates.pop();

      processCandidateNode(
          /* query = */ query, /* node = */ d_node.second,
          /* max_dist = */ max_dist, /* buffer_size = */ buffer_size,
          /* visited_set = */ visited_set,
          /* neighbors = */ neighbors, /* candidates = */ candidates);
    }

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ visited_set);

    return neighbors;
  }

  void processCandidateNode(const void *query, node_id_t &node, float &max_dist,
                            const int buffer_size, VisitedSet *visited_set,
                            PriorityQueue &neighbors,
                            PriorityQueue &candidates) {
    // Lock all operations on this specific node
    std::unique_lock<std::mutex> lock(_node_links_mutexes[node]);
    float dist = 0.f;

    node_id_t *neighbor_node_links = getNodeLinks(node);
    auto M = _parameter_config->max_edges_per_node;
    for (uint32_t i = 0; i < M; i++) {
      node_id_t neighbor_node_id = neighbor_node_links[i];

      // If using SSE, prefetch the next neighbor node data and the visited
      // marker
#ifdef USE_SSE
      if (i != M - 1) {
        _mm_prefetch(getNodeData(neighbor_node_links[i + 1]), _MM_HINT_T0);
        visited_set->prefetch(neighbor_node_links[i + 1]);
      }
#endif

      bool neighbor_is_visited =
          visited_set->isVisited(/* num = */ neighbor_node_id);

      if (neighbor_is_visited) {
        continue;
      }
      visited_set->insert(/* num = */ neighbor_node_id);
      dist = _distance->distance(
          /* x = */ query,
          /* y = */ getNodeData(neighbor_node_id),
          /* asymmetric = */ true);

      if (neighbors.size() < buffer_size || dist < max_dist) {
        candidates.emplace(-dist, neighbor_node_id);
        neighbors.emplace(dist, neighbor_node_id);
#ifdef USE_SSE
        _mm_prefetch(getNodeData(candidates.top().second), _MM_HINT_T0);
#endif
        if (neighbors.size() > buffer_size) {
          neighbors.pop();
        }
        if (!neighbors.empty()) {
          max_dist = neighbors.top().first;
        }
      }
    }
  }

  /**
   * @brief Selects neighbors from the PriorityQueue, according to the HNSW
   * heuristic. The neighbors priority queue contains elements sorted by
   * distance where the top element is the furthest neighbor from the query.
   */
  void selectNeighbors(PriorityQueue &neighbors) {
    if (neighbors.size() < _parameter_config->max_edges_per_node) {
      return;
    }

    PriorityQueue candidates;
    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(_parameter_config->max_edges_per_node);

    while (neighbors.size() > 0) {
      candidates.emplace(-neighbors.top().first, neighbors.top().second);
      neighbors.pop();
    }

    float cur_dist = 0.0;
    while (candidates.size() > 0) {
      if (saved_candidates.size() >= _parameter_config->max_edges_per_node) {
        break;
      }
      // Extract the closest element from candidates.
      dist_node_t current_pair = candidates.top();
      candidates.pop();

      bool should_keep_candidate = true;
      for (const dist_node_t &second_pair : saved_candidates) {

        cur_dist = _distance->distance(
            /* x = */ getNodeData(second_pair.second),
            /* y = */ getNodeData(current_pair.second));

        if (cur_dist < (-current_pair.first)) {
          should_keep_candidate = false;
          break;
        }
      }
      if (should_keep_candidate) {
        // We could do neighbors.emplace except we have to iterate
        // through saved_candidates, and std::priority_queue doesn't
        // support iteration (there is no technical reason why not).
        saved_candidates.push_back(current_pair);
      }
    }
    // TODO: implement my own priority queue, get rid of vector
    // saved_candidates, add directly to neighborqueue earlier.
    for (const dist_node_t &current_pair : saved_candidates) {
      neighbors.emplace(-current_pair.first, current_pair.second);
    }
  }

  void connectNeighbors(PriorityQueue &neighbors, node_id_t new_node_id) {
    // connects neighbors according to the HSNW heuristic

    // Lock all operations on this node
    std::unique_lock<std::mutex> lock(_node_links_mutexes[new_node_id]);

    node_id_t *new_node_links = getNodeLinks(new_node_id);
    int i = 0; // iterates through links for "new_node_id"

    while (neighbors.size() > 0) {
      node_id_t neighbor_node_id = neighbors.top().second;
      // add link to the current new node
      new_node_links[i] = neighbor_node_id;
      // now do the back-connections (a little tricky)

      std::unique_lock<std::mutex> neighbor_lock(
          _node_links_mutexes[neighbor_node_id]);
      node_id_t *neighbor_node_links = getNodeLinks(neighbor_node_id);
      bool is_inserted = false;
      for (size_t j = 0; j < _parameter_config->max_edges_per_node; j++) {
        if (neighbor_node_links[j] == neighbor_node_id) {
          // If there is a self-loop, replace the self-loop with
          // the desired link.
          neighbor_node_links[j] = new_node_id;
          is_inserted = true;
          break;
        }
      }
      if (!is_inserted) {
        // now, we may to replace one of the links. This will disconnect
        // the old neighbor and create a directed edge, so we have to be
        // very careful. To ensure we respect the pruning heuristic, we
        // construct a candidate set including the old links AND our new
        // one, then prune this candidate set to get the new neighbors.

        float max_dist = _distance->distance(
            /* x = */ getNodeData(neighbor_node_id),
            /* y = */ getNodeData(new_node_id));

        PriorityQueue candidates;
        candidates.emplace(max_dist, new_node_id);
        for (size_t j = 0; j < _parameter_config->max_edges_per_node; j++) {
          if (neighbor_node_links[j] != neighbor_node_id) {
            auto label = neighbor_node_links[j];
            auto distance = _distance->distance(
                /* x = */ getNodeData(neighbor_node_id),
                /* y = */ getNodeData(label));
            candidates.emplace(distance, label);
          }
        }
        selectNeighbors(candidates);
        // connect the pruned set of candidates, including self-loops:
        size_t j = 0;
        while (candidates.size() > 0) { // candidates
          neighbor_node_links[j] = candidates.top().second;
          candidates.pop();
          j++;
        }
        while (
            j <
            _parameter_config->max_edges_per_node) { // self-loops (unused links)
          neighbor_node_links[j] = neighbor_node_id;
          j++;
        }
      }

      // Unlock the current node we are iterating over
      neighbor_lock.unlock();

      // loop increments:
      i++;
      if (i >= _parameter_config->max_edges_per_node) {
        i = _parameter_config->max_edges_per_node;
      }
      neighbors.pop();
    }
  }

  /**
   * @brief Selects a node to use as the entry point for a new node.
   * This proceeds in a greedy fashion, by selecting the node with
   * the smallest distance to the query.
   *
   * @param query
   * @param num_initializations
   * @return node_id_t
   */
  inline node_id_t initializeSearch(const void *query,
                                    int num_initializations) {
    // select entry_node from a set of random entry point options
    if (num_initializations <= 0) {
      throw std::invalid_argument(
          "num_initializations must be greater than 0.");
    }

    int step_size = _cur_num_nodes / num_initializations;
    step_size = step_size ? step_size : 1;

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    for (node_id_t node = 0; node < _cur_num_nodes; node += step_size) {
      float dist = _distance->distance(
          /* x = */ query, /* y = */ getNodeData(node),
          /* asymmetric = */ true);
      if (dist < min_dist) {
        min_dist = dist;
        entry_node = node;
      }
    }
    return entry_node;
  }

  void relabel(const std::vector<node_id_t> &P) {
    // 1. Rewire all of the node connections
    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      node_id_t *links = getNodeLinks(n);
      for (int m = 0; m < _parameter_config->max_edges_per_node; m++) {
        links[m] = P[links[m]];
      }
    }

    // 2. Physically re-layout the nodes (in place)
    char *temp_data = new char[_data_size_bytes];
    node_id_t *temp_links = new node_id_t[_parameter_config->max_edges_per_node];
    label_t *temp_label = new label_t;

    auto *visited_set = _visited_set_pool->pollAvailableSet();

    // In this context, is_visited stores which nodes have been relocated
    // (it would be equivalent to name this variable "is_relocated").
    visited_set->clear();

    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      if (visited_set->isVisited(/* num = */ n)) {
        continue;
      }

      node_id_t src = n;
      node_id_t dest = P[src];

      // swap node at src with node at dest
      swapNodes(src, dest, temp_data, temp_links, temp_label);

      // mark src as having been relocated
      visited_set->insert(src);

      // recursively relocate the node from "dest"
      while (!visited_set->isVisited(/* num = */ dest)) {
        // mark node as having been relocated
        visited_set->insert(dest);
        // the value of src remains the same. However, dest needs
        // to change because the node located at src was previously
        // located at dest, and must be relocated to P[dest].
        dest = P[dest];

        // swap node at src with node at dest
        swapNodes(src, dest, temp_data, temp_links, temp_label);
      }
    }

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ visited_set);

    delete[] temp_data;
    delete[] temp_links;
    delete temp_label;
  }
};

} // namespace flatnav