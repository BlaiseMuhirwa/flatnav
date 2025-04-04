#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/index/Allocator.h>
#include <flatnav/index/PrimitiveTypes.h>
#include <flatnav/index/Pruning.h>
#include <flatnav/util/Datatype.h>
#include <flatnav/util/Macros.h>
#include <flatnav/util/Multithreading.h>
#include <flatnav/util/Reordering.h>
#include <flatnav/util/VisitedSetPool.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <thread>
#include <utility>
#include <vector>

using flatnav::distances::DistanceInterface;
using flatnav::util::DataType;
using flatnav::util::VisitedSet;
using flatnav::util::VisitedSetPool;

namespace flatnav {

template <typename dist_t, typename label_t>
class Index {
  using node_id_t = flatnav::index_node_id_t;
  using dist_label_t = flatnav::index_dist_label_t<label_t>;
  using dist_node_t = flatnav::index_dist_node_t;
  using PriorityQueue = flatnav::index_priority_queue_t;
  using MemoryAllocator = flatnav::FlatMemoryAllocator<label_t>;
  using IndexBuildParameters = flatnav::IndexBuildParameters;

  size_t _cur_num_nodes;
  std::unique_ptr<DistanceInterface<dist_t>> _distance;
  std::mutex _index_data_guard;
  uint32_t _num_threads;

  // Remembers which nodes we've visited, to avoid re-computing distances.
  VisitedSetPool* _visited_set_pool;
  std::vector<std::mutex> _node_links_mutexes;

  bool _collect_stats = false;

  MemoryAllocator& _memory_allocator;
  IndexBuildParameters* _params;
  PruningHeuristicSelector<dist_t, label_t> _ph_selector;

  // NOTE: These metrics are meaningful the most with single-threaded search.
  // With multi-threaded search, for instance, the number of distance computations will
  // accumulate across queries, which means at the end of the batched search, the number
  // you get is the cumulative sum of all distance computations across all queries.
  // Maybe that's what you want, but it's worth noting
  mutable std::atomic<uint64_t> _distance_computations = 0;
  mutable std::atomic<uint64_t> _metric_hops = 0;

  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;

  // A custom move constructor is needed because the class manages dynamic
  // resources (_index_memory, _visited_set_pool),
  // which require explicit ownership transfer and cleanup to avoid resource
  // leaks or double frees. The default move constructor cannot ensure these
  // resources are safely transferred and the source object is left in a valid
  // state.
  Index(Index&& other) noexcept
      : _cur_num_nodes(other._cur_num_nodes),
        _distance(std::move(other._distance)),
        _index_data_guard(std::move(other._index_data_guard)),
        _num_threads(other._num_threads),
        _visited_set_pool(std::exchange(other._visited_set_pool, nullptr)),
        _node_links_mutexes(std::move(other._node_links_mutexes)),
        _collect_stats(other._collect_stats),
        _memory_allocator(other._memory_allocator),  // reference copy
        _params(other._params),                      // pointer copy
        _ph_selector(other._memory_allocator,
                     *other._distance)  // reinit with moved distance
  {}

  Index& operator=(Index&& other) noexcept {
    if (this != &other) {
      delete _visited_set_pool;

      _cur_num_nodes = other._cur_num_nodes;
      _distance = std::move(other._distance);
      _index_data_guard = std::move(other._index_data_guard);
      _num_threads = other._num_threads;

      _visited_set_pool = std::exchange(other._visited_set_pool, nullptr);
      _node_links_mutexes = std::move(other._node_links_mutexes);

      _collect_stats = other._collect_stats;

      _memory_allocator = other._memory_allocator;  // reference copy
      _params = other._params;                      // pointer copy

      _ph_selector =
          PruningHeuristicSelector<dist_t, label_t>(_memory_allocator, *_distance);
    }
    return *this;
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_cur_num_nodes, *_distance);

    uint64_t total_mem = static_cast<uint64_t>(_params->node_size_bytes) *
                         static_cast<uint64_t>(_params->dataset_size);

    archive(cereal::binary_data(_memory_allocator.getIndexMemoryBlock(), total_mem));
  }

 public:
  /**
   * @brief Construct a new Index object for approximate near neighbor search.
   *
   * This constructor initializes an Index object with the specified distance
   * metric, dataset size, and maximum number of links per node. It also allows
   * for collecting statistics during the search process.
   *
   * @param dist The distance metric for the index. Options include l2
   * (euclidean) and inner product.
   * @param dataset_size The maximum number of vectors that can be inserted in
   * the index.
   * @param max_edges_per_node The maximum number of links per node.
   * @param collect_stats Flag indicating whether to collect statistics during
   * the search process.
   */
  Index(std::unique_ptr<DistanceInterface<dist_t>> dist,
        MemoryAllocator& memory_allocator, IndexBuildParameters* params,
        bool collect_stats = false)
      : _cur_num_nodes(0),
        _distance(std::move(dist)),
        _num_threads(1),
        _visited_set_pool(new VisitedSetPool(
            /* initial_pool_size = */ 1,
            /* num_elements = */ params->dataset_size)),
        _node_links_mutexes(params->dataset_size),
        _collect_stats(collect_stats),
        _memory_allocator(memory_allocator),
        _params(params),
        _ph_selector(memory_allocator, *_distance) {}

  ~Index() { delete _visited_set_pool; }

  void buildGraphLinks(const std::string& mtx_filename) {
    std::ifstream input_file(mtx_filename);
    if (!input_file.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + mtx_filename);
    }

    std::string line;
    // Skip the header
    while (std::getline(input_file, line)) {
      if (line[0] != '%')
        break;
    }

    std::istringstream iss(line);
    int num_vertices, num_edges;
    iss >> num_vertices >> num_vertices >> num_edges;

    // check that the number of vertices in the mtx file matches the number of
    // nodes in the index and that the number of edges is equal to the number of
    // links per node.
    if (num_vertices != _params->dataset_size) {
      throw std::runtime_error(
          "Number of vertices in the mtx file does not "
          "match the size allocated for the index.");
    }

    if (num_edges != _params->M) {
      throw std::runtime_error(
          "Number of edges in the mtx file does not match "
          "the number of links per node.");
    }

    int u, v;
    while (input_file >> u >> v) {
      // Adjust for 1-based indexing in Matrix Market format
      u--;
      v--;
      node_id_t* links = _memory_allocator.getNodeLinks(u);
      // Now add a directed edge from u to v. We need to check for the first
      // available slot in the links array since there might be other edges
      // added before this one. By definition, a slot is available if and only
      // if it points to the node itself.
      for (size_t i = 0; i < _params->M; i++) {
        if (links[i] == u) {
          links[i] = v;
          break;
        }
      }
    }

    input_file.close();
  }

  std::vector<std::vector<uint32_t>> getGraphOutdegreeTable() {
    std::vector<std::vector<uint32_t>> outdegree_table(_cur_num_nodes);
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      node_id_t* links = _memory_allocator.getNodeLinks(node);
      for (int i = 0; i < _params->M; i++) {
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
  void allocateNode(void* data, label_t& label, node_id_t& new_node_id) {
    new_node_id = _cur_num_nodes;
    _distance->transformData(
        /* destination = */ _memory_allocator.getNodeData(new_node_id),
        /* src = */ data);
    label_t* label_ptr = _memory_allocator.getNodeLabel(new_node_id);
    *label_ptr = label;
    node_id_t* links = _memory_allocator.getNodeLinks(new_node_id);
    // Initialize all edges to self
    std::fill_n(links, _params->M, new_node_id);
    _cur_num_nodes++;
  }

  /**
   * @brief Adds vectors to the index in batches.
   *
   * This method is responsible for adding vectors in batches, represented by
   * `data`, to the underlying graph. Each vector is associated with a label
   * provided in the `labels` vector. The method efficiently handles concurrent
   * additions by dividing the workload among multiple threads, defined by
   * `_num_threads`.
   *
   * The method ensures thread safety by employing locking mechanisms at the
   * node level in the underlying `connectNeighbors` and `beamSearch` methods.
   * This allows multiple threads to safely add vectors to the index without
   * causing data races or inconsistencies in the graph structure.
   *
   * @param data Pointer to the array of vectors to be added.
   * @param labels A vector of labels corresponding to each vector in `data`.
   * @param ef_construction Parameter for controlling the size of the dynamic
   * candidate list during the construction of the graph.
   * @param num_initializations Number of initializations for the search
   * algorithm. Must be greater than 0.
   *
   * @exception std::invalid_argument Thrown if `num_initializations` is less
   * than or equal to 0.
   * @exception std::runtime_error Thrown if the maximum number of nodes in the
   * index is reached.
   */
  template <typename data_type>
  void addBatch(void* data, std::vector<label_t>& labels,
                int num_initializations = 100) {
    if (num_initializations <= 0) {
      throw std::invalid_argument("num_initializations must be greater than 0.");
    }
    uint32_t total_num_nodes = labels.size();
    uint32_t data_dimension = _distance->dimension();
    int ef_construction = static_cast<int>(_params->ef_construction);

    // Don't spawn any threads if we are only using one.
    if (_num_threads == 1) {
      for (uint32_t row_index = 0; row_index < total_num_nodes; row_index++) {
        uint64_t offset =
            static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
        void* vector = (data_type*)data + offset;
        label_t label = labels[row_index];
        this->add(vector, label, ef_construction, num_initializations);
      }
      return;
    }
    
    flatnav::executeInParallel(
        /* start_index = */ 0, /* end_index = */ total_num_nodes,
        /* num_threads = */ _num_threads, /* function = */
        [&](uint32_t row_index) {
          uint64_t offset =
              static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
          void* vector = (data_type*)data + offset;
          label_t label = labels[row_index];
          this->add(vector, label, ef_construction, num_initializations);
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
  void add(void* data, label_t& label, int ef_construction, int num_initializations) {

    if (_cur_num_nodes >= _params->dataset_size) {
      throw std::runtime_error(
          "Maximum number of nodes reached. Consider "
          "increasing the `dataset_size` parameter to "
          "create a larger index.");
    }
    std::unique_lock<std::mutex> global_lock(_index_data_guard);
    auto entry_node = initializeSearch(data, num_initializations);
    node_id_t new_node_id;
    allocateNode(data, label, new_node_id);
    global_lock.unlock();

    if (new_node_id == 0) {
      return;
    }

    auto neighbors = beamSearch(
        /* query = */ data, /* entry_node = */ entry_node,
        /* buffer_size = */ ef_construction);

    int selection_M = std::max(static_cast<int>(_params->M / 2), 1);
    _ph_selector.select(_params->pruning_heuristic, neighbors, selection_M,
                        _params->pruning_heuristic_parameter);
    connectNeighbors(neighbors, new_node_id);
  }

  /***
   * @brief Search the index for the k nearest neighbors of the query.
   * @param query The query vector.
   * @param K The number of nearest neighbors to return.
   * @param ef_search The search beam width.
   * @param num_initializations The number of random initializations to use.
   */
  std::vector<dist_label_t> search(const void* query, const int K, int ef_search,
                                   int num_initializations = 100) {
    node_id_t entry_node = initializeSearch(query, num_initializations);
    PriorityQueue neighbors = beamSearch(/* query = */ query,
                                         /* entry_node = */ entry_node,
                                         /* buffer_size = */ std::max(ef_search, K));
    auto size = neighbors.size();
    std::vector<dist_label_t> results;
    results.reserve(size);
    while (!neighbors.empty()) {
      auto [distance, node_id] = neighbors.top();
      auto label = *_memory_allocator.getNodeLabel(node_id);
      results.emplace_back(distance, label);
      neighbors.pop();
    }
    std::sort(results.begin(), results.end(),
              [](const dist_label_t& left, const dist_label_t& right) {
                return left.first < right.first;
              });
    if (results.size() > static_cast<size_t>(K)) {
      results.resize(K);
    }

    return results;
  }

  void doGraphReordering(const std::vector<std::string>& reordering_methods) {

    for (const auto& method : reordering_methods) {
      auto outdegree_table = getGraphOutdegreeTable();
      std::vector<node_id_t> P;
      if (method == "gorder") {
        P = std::move(util::gOrder<node_id_t>(outdegree_table, 5));
      } else if (method == "rcm") {
        P = std::move(util::rcmOrder<node_id_t>(outdegree_table));
      } else {
        throw std::invalid_argument("Invalid reordering method: " + method);
      }

      relabel(P);
    }
  }

  void reorderGOrder(const int window_size = 5) {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P = util::gOrder<node_id_t>(outdegree_table, window_size);

    relabel(P);
  }

  void reorderRCM() {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P = util::rcmOrder<node_id_t>(outdegree_table);
    relabel(P);
  }

  static std::unique_ptr<Index<dist_t, label_t>> loadIndex(const std::string& filename,
                                                           MemoryAllocator& allocator,
                                                           IndexBuildParameters* params) {

    std::ifstream stream(filename, std::ios::binary);
    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }
    cereal::BinaryInputArchive archive(stream);

    // 1. Deserialize into temporary fields
    DataType data_type;
    size_t cur_num_nodes;

    std::unique_ptr<DistanceInterface<dist_t>> dist = std::make_unique<dist_t>();
    archive(cur_num_nodes, *dist);

    // 3. Deserialize memory into the existing allocator
    uint64_t mem_size = static_cast<uint64_t>(params->node_size_bytes) *
                        static_cast<uint64_t>(params->dataset_size);
    archive(cereal::binary_data(allocator.getIndexMemoryBlock(), mem_size));

    // 4. Construct the index with external allocator and params
    auto index = std::make_unique<Index<dist_t, label_t>>(std::move(dist), allocator,
                                                          params, false);
    // auto index =
    //     std::make_unique<Index<dist_t, label_t>>(std::move(dist), allocator, params);

    index->_cur_num_nodes = cur_num_nodes;
    index->_visited_set_pool = new VisitedSetPool(1, params->dataset_size);
    index->_num_threads = std::max<uint32_t>(1, std::thread::hardware_concurrency() / 2);
    index->_node_links_mutexes = std::vector<std::mutex>(params->dataset_size);
    return index;
  }

  void saveIndex(const std::string& filename) {
    std::ofstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    // Let's also save the metadata for the index
    _params->save(filename + ".metadata");

    cereal::BinaryOutputArchive archive(stream);
    archive(*this);
  }

  inline void setNumThreads(uint32_t num_threads) {
    if (num_threads == 0 || num_threads > std::thread::hardware_concurrency()) {
      throw std::invalid_argument(
          "Number of threads must be greater than 0 and less than or equal to "
          "the number of hardware threads.");
    }
    _num_threads = num_threads;
    if (_num_threads == 1) {
      _visited_set_pool->setPoolSize(1);
    }
  }

  inline uint64_t getTotalIndexMemory() const {
    return _memory_allocator.getTotalIndexMemory();
  }
  inline uint64_t mutexesAllocatedMemory() const {
    return static_cast<uint64_t>(_node_links_mutexes.size() * sizeof(std::mutex));
  }

  inline uint64_t visitedSetPoolAllocatedMemory() const {
    size_t pool_size = _visited_set_pool->poolSize();
    return static_cast<uint64_t>(pool_size * sizeof(VisitedSet));
  }

  inline uint32_t getNumThreads() const { return _num_threads; }

  inline size_t maxEdgesPerNode() const { return _params->M; }
  inline size_t dataSizeBytes() const { return _params->data_size_bytes; }

  inline size_t nodeSizeBytes() const { return _params->node_size_bytes; }

  inline size_t maxNodeCount() const { return _params->dataset_size; }

  inline size_t currentNumNodes() const { return _cur_num_nodes; }
  inline size_t dataDimension() const { return _distance->dimension(); }

  inline uint64_t distanceComputations() const { return _distance_computations.load(); }

  inline DataType getDataType() const { return _params->data_type; }

  void resetStats() {
    _distance_computations = 0;
    _metric_hops = 0;
  }

  void getIndexSummary() const {
    std::cout << "\nIndex Parameters\n" << std::flush;
    std::cout << "-----------------------------\n" << std::flush;
    std::cout << "max_edges_per_node (M): " << _params->M << "\n" << std::flush;
    std::cout << "data_size_bytes: " << _params->data_size_bytes << "\n" << std::flush;
    std::cout << "node_size_bytes: " << _params->node_size_bytes << "\n" << std::flush;
    std::cout << "dataset_size: " << _params->dataset_size << "\n" << std::flush;
    std::cout << "cur_num_nodes: " << _cur_num_nodes << "\n" << std::flush;

    _distance->getSummary();
  }

 private:
  friend class cereal::access;
  // Default constructor for cereal
  Index() = default;
  inline void swapNodes(node_id_t a, node_id_t b, void* temp_data, node_id_t* temp_links,
                        label_t* temp_label) {

    // stash b in temp
    std::memcpy(temp_data, _memory_allocator.getNodeData(b), _params->data_size_bytes);
    std::memcpy(temp_links, _memory_allocator.getNodeLinks(b),
                _params->M * sizeof(node_id_t));
    std::memcpy(temp_label, _memory_allocator.getNodeLabel(b), sizeof(label_t));

    // place node at a in b
    std::memcpy(_memory_allocator.getNodeData(b), _memory_allocator.getNodeData(a),
                _params->data_size_bytes);
    std::memcpy(_memory_allocator.getNodeLinks(b), _memory_allocator.getNodeLinks(a),
                _params->M * sizeof(node_id_t));
    std::memcpy(_memory_allocator.getNodeLabel(b), _memory_allocator.getNodeLabel(a),
                sizeof(label_t));

    // put node b in a
    std::memcpy(_memory_allocator.getNodeData(a), temp_data, _params->data_size_bytes);
    std::memcpy(_memory_allocator.getNodeLinks(a), temp_links,
                _params->M * sizeof(node_id_t));
    std::memcpy(_memory_allocator.getNodeLabel(a), temp_label, sizeof(label_t));
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

  PriorityQueue beamSearch(const void* query, const node_id_t entry_node,
                           const int buffer_size) {
    PriorityQueue neighbors;
    PriorityQueue candidates;

    auto* visited_set = _visited_set_pool->pollAvailableSet();
    visited_set->clear();

    // Prefetch the data for entry node before computing its distance.
#ifdef USE_SSE
    _mm_prefetch(_memory_allocator.getNodeData(entry_node), _MM_HINT_T0);
#endif

    float dist = _distance->distance(/* x = */ query,
                                     /* y = */ _memory_allocator.getNodeData(entry_node),
                                     /* asymmetric = */ true);

    float max_dist = dist;
    candidates.emplace(-dist, entry_node);
    neighbors.emplace(dist, entry_node);
    visited_set->insert(entry_node);

    while (!candidates.empty()) {
      auto [distance, node] = candidates.top();

      if (-distance > max_dist && neighbors.size() >= buffer_size) {
        break;
      }
      candidates.pop();

      // Prefetching the next candidate node data and visited set marker
      // before processing it. Note that this might not be useful if the current
      // iteration finds a neighbor that is closer than the current max
      // distance. In that case we would have prefetched data that is not used
      // immediately, but I think the cost of prefetching is low enough that
      // it's probably worth it.
#ifdef USE_SSE
      if (!candidates.empty()) {
        _mm_prefetch(_memory_allocator.getNodeData(candidates.top().second), _MM_HINT_T0);
        visited_set->prefetch(candidates.top().second);
      }
#endif

      processCandidateNode(
          /* query = */ query, /* node = */ node,
          /* max_dist = */ max_dist, /* buffer_size = */ buffer_size,
          /* visited_set = */ visited_set,
          /* neighbors = */ neighbors, /* candidates = */ candidates);
    }

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ visited_set);

    return neighbors;
  }

  void processCandidateNode(const void* query, node_id_t& node, float& max_dist,
                            const int buffer_size, VisitedSet* visited_set,
                            PriorityQueue& neighbors, PriorityQueue& candidates) {
    // Lock all operations on this specific node
    std::unique_lock<std::mutex> lock(_node_links_mutexes[node]);

    node_id_t* neighbor_node_links = _memory_allocator.getNodeLinks(node);
    for (uint32_t i = 0; i < _params->M; i++) {
      node_id_t neighbor_node_id = neighbor_node_links[i];

      // If using SSE, prefetch the next neighbor node data and the visited
      // marker
#ifdef USE_SSE
      if (i != _params->M - 1) {
        _mm_prefetch(_memory_allocator.getNodeData(neighbor_node_links[i + 1]),
                     _MM_HINT_T0);
        visited_set->prefetch(neighbor_node_links[i + 1]);
      }
#endif

      bool neighbor_is_visited = visited_set->isVisited(/* num = */ neighbor_node_id);

      if (neighbor_is_visited) {
        continue;
      }
      visited_set->insert(/* num = */ neighbor_node_id);
      float dist =
          _distance->distance(/* x = */ query,
                              /* y = */ _memory_allocator.getNodeData(neighbor_node_id),
                              /* asymmetric = */ true);

      if (_collect_stats) {
        _distance_computations.fetch_add(1);
      }

      if (neighbors.size() < buffer_size || dist < max_dist) {
        candidates.emplace(-dist, neighbor_node_id);
        neighbors.emplace(dist, neighbor_node_id);
#ifdef USE_SSE
        _mm_prefetch(_memory_allocator.getNodeData(candidates.top().second), _MM_HINT_T0);
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

  void connectNeighbors(PriorityQueue& neighbors, node_id_t new_node_id) {
    // connects neighbors according to the HSNW heuristic

    // Lock all operations on this node
    std::unique_lock<std::mutex> lock(_node_links_mutexes[new_node_id]);

    node_id_t* new_node_links = _memory_allocator.getNodeLinks(new_node_id);
    int i = 0;  // iterates through links for "new_node_id"

    while (neighbors.size() > 0) {
      node_id_t neighbor_node_id = neighbors.top().second;
      // add link to the current new node
      new_node_links[i] = neighbor_node_id;
      // now do the back-connections (a little tricky)

      std::unique_lock<std::mutex> neighbor_lock(_node_links_mutexes[neighbor_node_id]);
      node_id_t* neighbor_node_links = _memory_allocator.getNodeLinks(neighbor_node_id);
      bool is_inserted = false;
      for (size_t j = 0; j < _params->M; j++) {
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

        float max_dist =
            _distance->distance(/* x = */ _memory_allocator.getNodeData(neighbor_node_id),
                                /* y = */ _memory_allocator.getNodeData(new_node_id));

        PriorityQueue candidates;
        candidates.emplace(max_dist, new_node_id);
        for (size_t j = 0; j < _params->M; j++) {
          if (neighbor_node_links[j] != neighbor_node_id) {
            auto label = neighbor_node_links[j];
            auto distance = _distance->distance(
                /* x = */ _memory_allocator.getNodeData(neighbor_node_id),
                /* y = */ _memory_allocator.getNodeData(label));
            candidates.emplace(distance, label);
          }
        }
        // 2X larger than the previous call to selectNeighbors.
        _ph_selector.select(_params->pruning_heuristic, candidates, _params->M, 
                            _params->pruning_heuristic_parameter);
        // connect the pruned set of candidates, including self-loops:
        size_t j = 0;
        while (candidates.size() > 0) {  // candidates
          neighbor_node_links[j] = candidates.top().second;
          candidates.pop();
          j++;
        }
        while (j < _params->M) {  // self-loops (unused links)
          neighbor_node_links[j] = neighbor_node_id;
          j++;
        }
      }

      // Unlock the current node we are iterating over
      neighbor_lock.unlock();

      // loop increments:
      i++;
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
  inline node_id_t initializeSearch(const void* query, int num_initializations) {
    // select entry_node from a set of random entry point options
    if (num_initializations <= 0) {
      throw std::invalid_argument("num_initializations must be greater than 0.");
    }

    int step_size = _cur_num_nodes / num_initializations;
    step_size = step_size ? step_size : 1;

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    if (_collect_stats) {
      _distance_computations.fetch_add(num_initializations);
    }

    for (node_id_t node = 0; node < _cur_num_nodes; node += step_size) {
      float dist = _distance->distance(/* x = */ query,
                                       /* y = */ _memory_allocator.getNodeData(node),
                                       /* asymmetric = */ true);
      if (dist < min_dist) {
        min_dist = dist;
        entry_node = node;
      }
    }
    return entry_node;
  }

  void relabel(const std::vector<node_id_t>& P) {
    // 1. Rewire all of the node connections
    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      node_id_t* links = _memory_allocator.getNodeLinks(n);
      for (int m = 0; m < _params->M; m++) {
        links[m] = P[links[m]];
      }
    }

    // 2. Physically re-layout the nodes (in place)
    char* temp_data = new char[_params->data_size_bytes];
    node_id_t* temp_links = new node_id_t[_params->M];
    label_t* temp_label = new label_t;

    auto* visited_set = _visited_set_pool->pollAvailableSet();

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

}  // namespace flatnav
