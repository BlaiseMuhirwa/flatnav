#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/index/Index.h>
#include <flatnav/index/TwoPassStrategy.h>
#include <flatnav/util/Multithreading.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <vector>

using flatnav::distances::DistanceInterface;

namespace flatnav {

// Reduced num_initializations for Pass 2 since the graph already exists
// and we can leverage existing connectivity for entry point selection.
constexpr int PASS2_NUM_INITIALIZATIONS = 25;

/**
 * @brief Two-pass index construction for improved graph quality.
 *
 * This class implements a two-pass construction strategy where:
 * - Pass 1: Build a cheap graph (low M, low ef) and collect statistics
 * - Pass 2: Rebuild the graph using statistics to guide neighbor selection
 *
 * Three strategies are supported:
 * - HUBNESS_SCORING: Penalize nodes with high in-degree to prevent hub formation
 * - EDGE_QUALITY_SCORING: Score edges by distance and diversity, prefer quality edges
 * - INSERTION_ORDER_OPT: Reorder insertions so peripheral nodes are added first
 *
 * @tparam dist_t Distance function type (e.g., SquaredL2Distance)
 * @tparam label_t Label type for node metadata
 */
template <typename dist_t, typename label_t>
class TwoPassBuilder {
public:
  using IndexType = Index<dist_t, label_t>;
  using node_id_t = uint32_t;
  using dist_node_t = std::pair<float, node_id_t>;
  using PriorityQueue = typename IndexType::PriorityQueue;

private:
  std::unique_ptr<IndexType> _index;
  TwoPassConfig _config;
  size_t _dataset_size;
  size_t _dimension;

  // Statistics computed post-hoc after Pass 1 (strategy-specific)
  // Only the stats needed for the selected strategy are computed
  std::vector<uint32_t> _indegree_cache;        // For HUBNESS_SCORING
  std::vector<float> _node_avg_neighbor_dist;   // For EDGE_QUALITY_SCORING
  std::vector<float> _node_centrality;          // For INSERTION_ORDER_OPT

public:
  /**
   * @brief Construct a TwoPassBuilder.
   *
   * @param dist Distance function for the index
   * @param dataset_size Number of vectors to be indexed
   * @param config Two-pass construction configuration
   */
  TwoPassBuilder(std::unique_ptr<DistanceInterface<dist_t>> dist,
                 size_t dataset_size, const TwoPassConfig &config)
      : _config(config), _dataset_size(dataset_size),
        _dimension(dist->dimension()) {

    // Total M is the sum of Pass 1 and Pass 2 edges
    // Hypothesis: M_pass1 + M_pass2 achieves similar recall to single-pass M
    // but with faster construction due to cheaper graph maintenance
    int final_M = _config.M_pass1 + _config.M_pass2;

    _index = std::make_unique<IndexType>(
        std::move(dist), static_cast<int>(dataset_size), final_M,
        /* collect_stats = */ false);

    _index->setNumThreads(_config.num_threads);

  }

  /**
   * @brief Build the index using two-pass construction.
   *
   * @tparam data_type Data type of the vectors (float, int8_t, uint8_t)
   * @param data Pointer to the vector data
   * @param labels Vector of labels for each data point
   */
  template <typename data_type>
  void build(void *data, std::vector<label_t> &labels) {
    if (_config.strategy == TwoPassStrategy::NONE) {
      // Fall back to single-pass construction
      _index->template addBatch<data_type>(data, labels,
                                           _config.ef_construction_pass2,
                                           _config.num_initializations);
      return;
    }

    // Pass 1: Build cheap graph and collect statistics
    runPass1<data_type>(data, labels);
    collectStatistics();
    // Pass 2: Rebuild with strategy-specific modifications
    runPass2<data_type>(data);
  }

  /**
   * @brief Get the built index.
   */
  IndexType &getIndex() { return *_index; }

  /**
   * @brief Release ownership of the built index.
   */
  std::unique_ptr<IndexType> releaseIndex() { return std::move(_index); }

  /**
   * @brief Get the in-degree statistics collected during Pass 1.
   */
  const std::vector<uint32_t> &getIndegrees() const { return _indegree_cache; }

  /**
   * @brief Get the centrality scores computed from Pass 1.
   */
  const std::vector<float> &getCentralityScores() const {
    return _node_centrality;
  }

  /**
   * @brief Get the average neighbor distances from Pass 1.
   */
  const std::vector<float> &getAvgNeighborDistances() const {
    return _node_avg_neighbor_dist;
  }

private:
  /**
   * @brief Pass 1: Build a graph using only M_pass1 edges per node.
   *
   * The index is allocated with M_pass1 + M_pass2 slots, but Pass 1 only
   * fills the first M_pass1 slots. The remaining slots stay as self-loops
   * for Pass 2 to fill.
   */
  template <typename data_type> void runPass1(void *data, std::vector<label_t> &labels) {
    uint32_t total_num_nodes = labels.size();
    uint32_t data_dimension = _index->_distance->dimension();

    if (_config.num_threads == 1) {
      for (uint32_t row_index = 0; row_index < total_num_nodes; row_index++) {
        uint64_t offset = static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
        void *vector = reinterpret_cast<data_type *>(data) + offset;
        label_t label = labels[row_index];
        addNodePass1(vector, label, _config.ef_construction_pass1,
                     _config.num_initializations);
      }
    } else {
      flatnav::executeInParallel(
          0, total_num_nodes, _config.num_threads,
          [&](uint32_t row_index) {
            uint64_t offset = static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
            void *vector = reinterpret_cast<data_type *>(data) + offset;
            label_t label = labels[row_index];
            addNodePass1(vector, label, _config.ef_construction_pass1,
                         _config.num_initializations);
          });
    }
  }

  /**
   * @brief Add a node during Pass 1, using only M_pass1 edges.
   */
  void addNodePass1(void *data, label_t label, int ef_construction,
                    int num_initializations) {
    if (_index->_cur_num_nodes >= _index->_max_node_count) {
      throw std::runtime_error("Maximum number of nodes reached.");
    }

    std::unique_lock<std::mutex> global_lock(_index->_index_data_guard);
    auto entry_node = _index->initializeSearch(data, num_initializations);
    node_id_t new_node_id;
    _index->allocateNode(data, label, new_node_id);
    global_lock.unlock();

    if (new_node_id == 0) {
      return;
    }

    auto neighbors = _index->beamSearch(data, entry_node, ef_construction);

    // Use M_pass1 for neighbor selection (not full M)
    int selection_M = std::max(_config.M_pass1 / 2, 1);
    _index->selectNeighbors(neighbors, selection_M);

    // Connect using only M_pass1 slots
    connectNeighborsPass1(neighbors, new_node_id);
  }

  /**
   * @brief Connect neighbors using only M_pass1 slots.
   *
   * Similar to Index::connectNeighbors but limits connections to M_pass1
   * slots, leaving the remaining slots as self-loops for Pass 2.
   * Uses ordered locking (lower node_id first) to prevent deadlock.
   *
   * No inline stats collection - statistics are computed post-hoc in
   * collectStatistics() for better performance at scale.
   */
  void connectNeighborsPass1(PriorityQueue &neighbors, node_id_t new_node_id) {
    node_id_t *new_node_links = _index->getNodeLinks(new_node_id);
    int i = 0;

    while (neighbors.size() > 0 && i < _config.M_pass1) {
      node_id_t neighbor_node_id = neighbors.top().second;

      // Lock in consistent order (lower node_id first) to prevent deadlock
      node_id_t first_lock = std::min(new_node_id, neighbor_node_id);
      node_id_t second_lock = std::max(new_node_id, neighbor_node_id);

      std::unique_lock<std::mutex> lock1(_index->_node_links_mutexes[first_lock]);
      std::unique_lock<std::mutex> lock2(_index->_node_links_mutexes[second_lock],
                                          std::defer_lock);
      if (first_lock != second_lock) {
        lock2.lock();
      }

      new_node_links[i] = neighbor_node_id;

      node_id_t *neighbor_node_links = _index->getNodeLinks(neighbor_node_id);

      bool is_inserted = false;
      // Only check first M_pass1 slots for back-connection
      for (int j = 0; j < _config.M_pass1; j++) {
        if (neighbor_node_links[j] == neighbor_node_id) {
          neighbor_node_links[j] = new_node_id;
          is_inserted = true;
          break;
        }
      }

      if (!is_inserted) {
        // Prune and re-select within M_pass1 slots
        float max_dist = _index->_distance->distance(
            _index->getNodeData(neighbor_node_id),
            _index->getNodeData(new_node_id));

        PriorityQueue candidates;
        candidates.emplace(max_dist, new_node_id);
        for (int j = 0; j < _config.M_pass1; j++) {
          if (neighbor_node_links[j] != neighbor_node_id) {
            auto neighbor = neighbor_node_links[j];
            auto distance = _index->_distance->distance(
                _index->getNodeData(neighbor_node_id),
                _index->getNodeData(neighbor));
            candidates.emplace(distance, neighbor);
          }
        }

        _index->selectNeighbors(candidates, _config.M_pass1);

        int j = 0;
        while (candidates.size() > 0 && j < _config.M_pass1) {
          neighbor_node_links[j] = candidates.top().second;
          candidates.pop();
          j++;
        }
        while (j < _config.M_pass1) {
          neighbor_node_links[j] = neighbor_node_id;
          j++;
        }
      }

      i++;
      neighbors.pop();
    }
  }

  /**
   * @brief Compute statistics from Pass 1 graph (post-hoc).
   *
   * Only computes the statistics needed for the selected strategy:
   * - HUBNESS_SCORING: in-degree only
   * - EDGE_QUALITY_SCORING: average neighbor distance only
   * - INSERTION_ORDER_OPT: centrality (in-degree + out-degree)
   *
   * All computations are parallelized. This is much faster than inline
   * atomic collection during the hot construction path.
   */
  void collectStatistics() {
    uint32_t num_nodes = _index->currentNumNodes();

    switch (_config.strategy) {
    case TwoPassStrategy::HUBNESS_SCORING:
      computeIndegrees(num_nodes);
      break;

    case TwoPassStrategy::EDGE_QUALITY_SCORING:
      computeAvgNeighborDistances(num_nodes);
      break;

    case TwoPassStrategy::INSERTION_ORDER_OPT:
      computeIndegrees(num_nodes);
      computeCentralityScores(num_nodes);
      break;

    case TwoPassStrategy::RE_PRUNE_FULL:
      // Collect hubness stats for guiding the full re-pruning
      computeIndegrees(num_nodes);
      break;

    default:
      // NONE strategy - no stats needed
      break;
    }
  }

  /**
   * @brief Compute in-degree for each node by scanning all edges.
   *
   * Parallelized O(N * M_pass1) scan. Uses atomics for the accumulation
   * since this is a one-time post-hoc pass (not hot construction path).
   */
  void computeIndegrees(uint32_t num_nodes) {
    // Use atomic array for parallel accumulation
    auto indegree_atomic = std::make_unique<std::atomic<uint32_t>[]>(num_nodes);
    for (uint32_t i = 0; i < num_nodes; i++) {
      indegree_atomic[i].store(0, std::memory_order_relaxed);
    }

    // Each thread scans its assigned nodes and increments neighbor in-degrees
    auto countEdges = [&](uint32_t n) {
      node_id_t *links = _index->getNodeLinks(n);
      for (int j = 0; j < _config.M_pass1; j++) {
        node_id_t neighbor = links[j];
        if (neighbor != n && neighbor < num_nodes) {
          indegree_atomic[neighbor].fetch_add(1, std::memory_order_relaxed);
        }
      }
    };

    if (_config.num_threads > 1) {
      flatnav::executeInParallel(0, num_nodes, _config.num_threads, countEdges);
    } else {
      for (uint32_t n = 0; n < num_nodes; n++) {
        countEdges(n);
      }
    }

    // Copy to non-atomic cache for fast access in Pass 2
    _indegree_cache.resize(num_nodes);
    auto copyToCache = [&](uint32_t n) {
      _indegree_cache[n] = indegree_atomic[n].load(std::memory_order_relaxed);
    };

    if (_config.num_threads > 1) {
      flatnav::executeInParallel(0, num_nodes, _config.num_threads, copyToCache);
    } else {
      for (uint32_t n = 0; n < num_nodes; n++) {
        copyToCache(n);
      }
    }
  }

  /**
   * @brief Compute average neighbor distance for each node.
   *
   * Parallelized O(N * M_pass1) scan with distance computations.
   * Each node's computation is independent - no atomics needed.
   */
  void computeAvgNeighborDistances(uint32_t num_nodes) {
    _node_avg_neighbor_dist.resize(num_nodes, 0.0f);

    auto computeNodeAvgDist = [&](uint32_t n) {
      node_id_t *links = _index->getNodeLinks(n);
      float sum_dist = 0.0f;
      int edge_count = 0;

      for (int j = 0; j < _config.M_pass1; j++) {
        node_id_t neighbor = links[j];
        if (neighbor != n) {
          float dist = _index->_distance->distance(
              _index->getNodeData(n), _index->getNodeData(neighbor));
          sum_dist += dist;
          edge_count++;
        }
      }

      if (edge_count > 0) {
        _node_avg_neighbor_dist[n] = sum_dist / static_cast<float>(edge_count);
      }
    };

    if (_config.num_threads > 1) {
      flatnav::executeInParallel(0, num_nodes, _config.num_threads, computeNodeAvgDist);
    } else {
      for (uint32_t n = 0; n < num_nodes; n++) {
        computeNodeAvgDist(n);
      }
    }
  }

  /**
   * @brief Compute centrality scores based on degree.
   *
   * Requires computeIndegrees() to have been called first.
   * Centrality = in-degree + out-degree
   */
  void computeCentralityScores(uint32_t num_nodes) {
    _node_centrality.resize(num_nodes, 0.0f);

    auto computeNodeCentrality = [&](uint32_t n) {
      // Out-degree: count non-self-loop edges in M_pass1 slots
      node_id_t *links = _index->getNodeLinks(n);
      uint32_t outdegree = 0;
      for (int j = 0; j < _config.M_pass1; j++) {
        if (links[j] != n) {
          outdegree++;
        }
      }

      // Centrality = in-degree + out-degree
      _node_centrality[n] = static_cast<float>(_indegree_cache[n] + outdegree);
    };

    if (_config.num_threads > 1) {
      flatnav::executeInParallel(0, num_nodes, _config.num_threads, computeNodeCentrality);
    } else {
      for (uint32_t n = 0; n < num_nodes; n++) {
        computeNodeCentrality(n);
      }
    }
  }

  /**
   * @brief Get insertion order sorted by centrality (ascending).
   */
  std::vector<uint32_t> getInsertionOrder() const {
    size_t num_nodes = _index->currentNumNodes();
    std::vector<uint32_t> order(num_nodes);
    std::iota(order.begin(), order.end(), 0);

    // Sort by centrality ascending (peripheral nodes first)
    std::sort(order.begin(), order.end(),
              [this](uint32_t a, uint32_t b) {
                return _node_centrality[a] < _node_centrality[b];
              });

    return order;
  }

  /**
   * @brief Pass 2: Add M_pass2 edges to each node's remaining slots.
   *
   * Unlike the old approach that reset and rebuilt the graph, this adds
   * edges to the remaining slots (M_pass1 to M_pass1+M_pass2-1) while
   * preserving Pass 1 connectivity and applying hub penalties.
   *
   * For RE_PRUNE_FULL strategy, this instead replaces ALL edges using
   * the Pass 1 graph only for navigation.
   */
  template <typename data_type> void runPass2(void *data) {
    // Dispatch to re-pruning method for RE_PRUNE_FULL strategy
    if (_config.strategy == TwoPassStrategy::RE_PRUNE_FULL) {
      runPass2RePruneFull<data_type>(data);
      return;
    }

    uint32_t total_num_nodes = _index->currentNumNodes();
    uint32_t data_dimension = _index->_distance->dimension();

    if (_config.num_threads == 1) {
      for (uint32_t node_id = 0; node_id < total_num_nodes; node_id++) {
        uint64_t offset = static_cast<uint64_t>(node_id) * static_cast<uint64_t>(data_dimension);
        void *vector = reinterpret_cast<data_type *>(data) + offset;
        addEdgesPass2(vector, node_id);
      }
    } else {
      flatnav::executeInParallel(
          0, total_num_nodes, _config.num_threads,
          [&](uint32_t node_id) {
            uint64_t offset = static_cast<uint64_t>(node_id) * static_cast<uint64_t>(data_dimension);
            void *vector = reinterpret_cast<data_type *>(data) + offset;
            addEdgesPass2(vector, node_id);
          });
    }
  }

  /**
   * @brief Add M_pass2 edges to a node's remaining slots.
   *
   * Dispatches to the appropriate candidate finding method based on config.
   */
  void addEdgesPass2(void *data, node_id_t node_id) {
    if (_config.pass2_candidate_method == Pass2CandidateMethod::NEIGHBOR_EXPANSION) {
      addEdgesPass2NeighborExpansion(data, node_id);
    } else {
      addEdgesPass2BeamSearch(data, node_id);
    }
  }

  /**
   * @brief Add M_pass2 edges using beam search (original method).
   *
   * Uses beam search on the existing Pass 1 graph to find candidates,
   * then selects M_pass2 neighbors with hub penalty applied.
   * More thorough but O(ef * log N) per node.
   */
  void addEdgesPass2BeamSearch(void *data, node_id_t node_id) {
    // Use existing Pass 1 neighbors as entry point (more efficient than random)
    node_id_t *links = _index->getNodeLinks(node_id);
    node_id_t entry_node = links[0]; // First Pass 1 neighbor
    if (entry_node == node_id) {
      // Fallback to random if no Pass 1 neighbors
      entry_node = _index->initializeSearch(data, PASS2_NUM_INITIALIZATIONS);
    }

    // Search on existing graph
    auto candidates = _index->beamSearch(data, entry_node,
                                         _config.ef_construction_pass2);

    // Remove self and existing Pass 1 neighbors from candidates
    PriorityQueue filtered_candidates;
    std::unordered_set<node_id_t> existing_neighbors;
    existing_neighbors.insert(node_id); // Exclude self
    for (int j = 0; j < _config.M_pass1; j++) {
      if (links[j] != node_id) {
        existing_neighbors.insert(links[j]);
      }
    }

    while (candidates.size() > 0) {
      auto [dist, candidate_id] = candidates.top();
      candidates.pop();
      if (existing_neighbors.find(candidate_id) == existing_neighbors.end()) {
        filtered_candidates.emplace(dist, candidate_id);
      }
    }

    // Select M_pass2 neighbors with hub penalty
    selectNeighborsWithStrategy(filtered_candidates, _config.M_pass2, node_id);

    // Connect to remaining slots (M_pass1 to M_pass1+M_pass2-1)
    connectNeighborsPass2(filtered_candidates, node_id);
  }

  /**
   * @brief Add M_pass2 edges using neighbor-of-neighbor expansion.
   *
   * Instead of expensive beam search, explores the 2-hop or 3-hop
   * neighborhood of existing neighbors. This is O(M^2) or O(M^3) per node
   * instead of O(ef * log N), which is much faster at 100M+ scale.
   *
   * Uses the existing VisitedSetPool for O(1) exclusion lookups without
   * allocating new memory for each node.
   */
  void addEdgesPass2NeighborExpansion(void *data, node_id_t node_id) {
    node_id_t *links = _index->getNodeLinks(node_id);

    // Collect existing Pass 1 neighbors
    std::vector<node_id_t> pass1_neighbors;
    pass1_neighbors.reserve(_config.M_pass1);
    for (int j = 0; j < _config.M_pass1; j++) {
      if (links[j] != node_id) {
        pass1_neighbors.push_back(links[j]);
      }
    }

    // If no Pass 1 neighbors, fall back to beam search
    if (pass1_neighbors.empty()) {
      addEdgesPass2BeamSearch(data, node_id);
      return;
    }

    // Use VisitedSet from pool for O(1) exclusion lookup without allocation
    auto *visited = _index->_visited_set_pool->pollAvailableSet();
    visited->clear();

    // Mark self and existing neighbors as visited (excluded)
    visited->insert(node_id);
    for (node_id_t neighbor : pass1_neighbors) {
      visited->insert(neighbor);
    }

    PriorityQueue candidates;
    int total_M = _config.M_pass1 + _config.M_pass2;

    // 2-hop: neighbors of neighbors
    std::vector<node_id_t> hop2_nodes;
    for (node_id_t neighbor : pass1_neighbors) {
      node_id_t *neighbor_links = _index->getNodeLinks(neighbor);
      for (int j = 0; j < total_M; j++) {
        node_id_t candidate = neighbor_links[j];
        if (candidate != neighbor && !visited->isVisited(candidate)) {
          visited->insert(candidate);
          float dist = _index->_distance->distance(data, _index->getNodeData(candidate));
          candidates.emplace(dist, candidate);

          // Save for potential 3-hop exploration
          if (_config.neighbor_expansion_hops >= 3) {
            hop2_nodes.push_back(candidate);
          }
        }
      }
    }

    // 3-hop: neighbors of neighbors of neighbors (if configured)
    if (_config.neighbor_expansion_hops >= 3) {
      for (node_id_t hop2_node : hop2_nodes) {
        node_id_t *hop2_links = _index->getNodeLinks(hop2_node);
        for (int j = 0; j < total_M; j++) {
          node_id_t candidate = hop2_links[j];
          if (candidate != hop2_node && !visited->isVisited(candidate)) {
            visited->insert(candidate);
            float dist = _index->_distance->distance(data, _index->getNodeData(candidate));
            candidates.emplace(dist, candidate);
          }
        }
      }
    }

    // Return visited set to pool before further processing
    _index->_visited_set_pool->pushVisitedSet(visited);

    // Select M_pass2 neighbors with hub penalty
    selectNeighborsWithStrategy(candidates, _config.M_pass2, node_id);

    // Connect to remaining slots
    connectNeighborsPass2(candidates, node_id);
  }

  /**
   * @brief Connect Pass 2 neighbors to slots M_pass1 through M_pass1+M_pass2-1.
   *
   * Uses ordered locking (lower node_id first) to prevent deadlock.
   */
  void connectNeighborsPass2(PriorityQueue &neighbors, node_id_t node_id) {
    node_id_t *node_links = _index->getNodeLinks(node_id);
    int slot = _config.M_pass1; // Start after Pass 1 slots

    while (neighbors.size() > 0 && slot < _config.M_pass1 + _config.M_pass2) {
      node_id_t neighbor_id = neighbors.top().second;

      // Lock in consistent order (lower node_id first) to prevent deadlock
      node_id_t first_lock = std::min(node_id, neighbor_id);
      node_id_t second_lock = std::max(node_id, neighbor_id);

      std::unique_lock<std::mutex> lock1(_index->_node_links_mutexes[first_lock]);
      std::unique_lock<std::mutex> lock2(_index->_node_links_mutexes[second_lock],
                                          std::defer_lock);
      if (first_lock != second_lock) {
        lock2.lock();
      }

      // Add forward edge
      node_links[slot] = neighbor_id;

      // Back-connection: try to add to neighbor's Pass 2 slots
      node_id_t *neighbor_links = _index->getNodeLinks(neighbor_id);
      for (int j = _config.M_pass1; j < _config.M_pass1 + _config.M_pass2; j++) {
        if (neighbor_links[j] == neighbor_id) { // Empty slot (self-loop)
          neighbor_links[j] = node_id;
          break;
        }
      }

      slot++;
      neighbors.pop();
    }
  }

  /**
   * @brief Pass 2 with full re-pruning: Replace ALL edges using statistics.
   *
   * Unlike additive strategies that preserve Pass 1 edges, this method:
   * 1. Uses Pass 1 graph ONLY for navigation (neighbor expansion)
   * 2. Selects FULL M neighbors with hubness-guided pruning
   * 3. REPLACES all edges, allowing suboptimal Pass 1 edges to be reconsidered
   * 4. Adds back-connections to ensure bidirectional navigability
   *
   * This should achieve recall comparable to baseline while being faster
   * because Pass 1 can use very low ef_construction.
   */
  template <typename data_type> void runPass2RePruneFull(void *data) {
    uint32_t total_num_nodes = _index->currentNumNodes();
    uint32_t data_dimension = _index->_distance->dimension();
    int full_M = _config.M_pass1 + _config.M_pass2;

    // Phase 1: Each node selects its new neighbors (parallel, no conflicts)
    // Store the new edges temporarily
    std::vector<std::vector<node_id_t>> new_edges(total_num_nodes);

    auto selectNewNeighbors = [&](uint32_t node_id) {
      uint64_t offset =
          static_cast<uint64_t>(node_id) * static_cast<uint64_t>(data_dimension);
      void *node_data = reinterpret_cast<data_type *>(data) + offset;

      // Find candidates using neighbor expansion on Pass 1 graph
      PriorityQueue candidates =
          findCandidatesForRePrune(node_data, node_id, full_M);

      if (candidates.size() == 0) {
        // Keep existing edges
        node_id_t *links = _index->getNodeLinks(node_id);
        for (int j = 0; j < full_M; j++) {
          if (links[j] != node_id) {
            new_edges[node_id].push_back(links[j]);
          }
        }
        return;
      }

      // Select full M neighbors with hubness-guided pruning
      selectNeighborsForRePrune(candidates, full_M, node_id);

      // Store selected neighbors
      new_edges[node_id].reserve(candidates.size());
      while (candidates.size() > 0) {
        new_edges[node_id].push_back(candidates.top().second);
        candidates.pop();
      }
    };

    if (_config.num_threads == 1) {
      for (uint32_t node_id = 0; node_id < total_num_nodes; node_id++) {
        selectNewNeighbors(node_id);
      }
    } else {
      flatnav::executeInParallel(0, total_num_nodes, _config.num_threads,
                                 selectNewNeighbors);
    }

    // Phase 2: Write forward edges (parallel - each node writes to its own slots)
    auto writeForwardEdges = [&](uint32_t node_id) {
      node_id_t *links = _index->getNodeLinks(node_id);
      int slot = 0;
      for (node_id_t neighbor_id : new_edges[node_id]) {
        if (slot >= full_M) break;
        links[slot++] = neighbor_id;
      }
      // Fill remaining slots with self-loops
      while (slot < full_M) {
        links[slot++] = node_id;
      }
    };

    if (_config.num_threads == 1) {
      for (uint32_t node_id = 0; node_id < total_num_nodes; node_id++) {
        writeForwardEdges(node_id);
      }
    } else {
      flatnav::executeInParallel(0, total_num_nodes, _config.num_threads,
                                 writeForwardEdges);
    }

    // Phase 3: Add back-connections (parallel with per-node locking)
    auto addBackConnections = [&](uint32_t node_id) {
      for (node_id_t neighbor_id : new_edges[node_id]) {
        addBackConnectionLocked(node_id, neighbor_id, full_M);
      }
    };

    if (_config.num_threads == 1) {
      for (uint32_t node_id = 0; node_id < total_num_nodes; node_id++) {
        addBackConnections(node_id);
      }
    } else {
      flatnav::executeInParallel(0, total_num_nodes, _config.num_threads,
                                 addBackConnections);
    }
  }

  /**
   * @brief Add a back-connection from neighbor to node (thread-safe version).
   *
   * Uses mutex locking to allow parallel back-connection updates.
   * If neighbor has an empty slot, add the connection directly.
   * Otherwise, check if this connection improves the neighbor's edges.
   */
  void addBackConnectionLocked(node_id_t node_id, node_id_t neighbor_id, int full_M) {
    std::unique_lock<std::mutex> lock(_index->_node_links_mutexes[neighbor_id]);

    node_id_t *neighbor_links = _index->getNodeLinks(neighbor_id);

    // Check if connection already exists or find empty slot
    for (int j = 0; j < full_M; j++) {
      if (neighbor_links[j] == node_id) {
        return; // Already connected
      }
      if (neighbor_links[j] == neighbor_id) {
        // Found empty slot (self-loop), use it
        neighbor_links[j] = node_id;
        return;
      }
    }

    // No empty slot - need to decide if we should replace an existing edge
    // Use distance-based replacement: replace if this is closer than the worst edge
    float dist_to_node = _index->_distance->distance(
        _index->getNodeData(neighbor_id), _index->getNodeData(node_id));

    int worst_slot = -1;
    float worst_dist = dist_to_node;

    for (int j = 0; j < full_M; j++) {
      if (neighbor_links[j] != neighbor_id) {
        float dist = _index->_distance->distance(
            _index->getNodeData(neighbor_id),
            _index->getNodeData(neighbor_links[j]));
        if (dist > worst_dist) {
          worst_dist = dist;
          worst_slot = j;
        }
      }
    }

    if (worst_slot >= 0) {
      neighbor_links[worst_slot] = node_id;
    }
  }

  /**
   * @brief Find candidate neighbors for re-pruning using seeded beam search.
   *
   * Instead of pure neighbor expansion (which doesn't scale), this uses
   * beam search seeded from the node's Pass 1 neighbors. This explores
   * O(ef × log N) nodes instead of O(M²), making it effective at 100M+ scale.
   */
  PriorityQueue findCandidatesForRePrune(void *node_data, node_id_t node_id,
                                         int target_M) {
    node_id_t *links = _index->getNodeLinks(node_id);

    // Find the best Pass 1 neighbor to use as entry point
    node_id_t best_entry = node_id;
    float best_dist = std::numeric_limits<float>::max();

    for (int j = 0; j < _config.M_pass1; j++) {
      if (links[j] != node_id) {
        float dist = _index->_distance->distance(node_data,
                                                  _index->getNodeData(links[j]));
        if (dist < best_dist) {
          best_dist = dist;
          best_entry = links[j];
        }
      }
    }

    // If no Pass 1 neighbors, use random initialization
    if (best_entry == node_id) {
      best_entry = _index->initializeSearch(node_data, PASS2_NUM_INITIALIZATIONS);
    }

    // Use beam search with Pass 2 ef_construction to find candidates
    // This explores O(ef × log N) nodes, which scales to 100M+
    PriorityQueue candidates = _index->beamSearch(
        node_data, best_entry, _config.ef_construction_pass2);

    // Remove self from candidates if present
    PriorityQueue filtered;
    while (candidates.size() > 0) {
      auto [dist, id] = candidates.top();
      candidates.pop();
      if (id != node_id) {
        filtered.emplace(dist, id);
      }
    }

    return filtered;
  }

  /**
   * @brief Select neighbors for re-pruning with hubness-guided pruning.
   *
   * Uses the standard greedy pruning algorithm but applies a penalty to
   * high-hubness nodes, encouraging diverse connectivity.
   */
  void selectNeighborsForRePrune(PriorityQueue &neighbors, int M,
                                 node_id_t node_id) {
    if (neighbors.size() <= static_cast<size_t>(M)) {
      return;
    }

    std::priority_queue<std::pair<float, node_id_t>> candidates;
    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);

    // Convert max-heap to min-heap by negating distances
    while (neighbors.size() > 0) {
      auto [distance, id] = neighbors.top();
      candidates.emplace(-distance, id);
      neighbors.pop();
    }

    while (candidates.size() > 0) {
      if (static_cast<int>(saved_candidates.size()) >= M) {
        break;
      }

      auto [neg_distance, current_node_id] = candidates.top();
      float distance_to_query = -neg_distance;
      candidates.pop();

      bool should_keep = true;

      for (const auto &[saved_dist, saved_node_id] : saved_candidates) {
        float cur_dist = _index->_distance->distance(
            _index->getNodeData(saved_node_id),
            _index->getNodeData(current_node_id));

        // Apply hubness penalty to the threshold
        float threshold = distance_to_query;
        if (!_indegree_cache.empty()) {
          threshold = applyHubnessPenalty(threshold, current_node_id, saved_node_id);
        }

        if (cur_dist < threshold) {
          should_keep = false;
          break;
        }
      }

      if (should_keep) {
        saved_candidates.push_back({distance_to_query, current_node_id});
      }
    }

    // Reconstruct neighbors queue
    for (const auto &[dist, id] : saved_candidates) {
      neighbors.emplace(dist, id);
    }
  }

  /**
   * @brief Replace all edges for a node with the selected neighbors.
   *
   * Unlike connectNeighborsPass2 which only fills Pass 2 slots, this
   * replaces ALL M slots. No back-connection is done to avoid conflicts
   * during parallel re-pruning (each node handles its own edges).
   */
  void replaceAllEdges(PriorityQueue &neighbors, node_id_t node_id, int full_M) {
    std::unique_lock<std::mutex> lock(_index->_node_links_mutexes[node_id]);

    node_id_t *links = _index->getNodeLinks(node_id);
    int slot = 0;

    while (neighbors.size() > 0 && slot < full_M) {
      links[slot++] = neighbors.top().second;
      neighbors.pop();
    }

    // Fill remaining slots with self-loops
    while (slot < full_M) {
      links[slot++] = node_id;
    }
  }

  /**
   * @brief Strategy-modified neighbor selection.
   */
  void selectNeighborsWithStrategy(PriorityQueue &neighbors, int M,
                                   node_id_t new_node_id) {
    if (neighbors.size() < static_cast<size_t>(M)) {
      return;
    }

    std::priority_queue<std::pair<float, node_id_t>> candidates;
    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);

    // Convert max-heap to min-heap by negating distances
    while (neighbors.size() > 0) {
      auto [distance, id] = neighbors.top();
      candidates.emplace(-distance, id);
      neighbors.pop();
    }

    while (candidates.size() > 0) {
      if (static_cast<int>(saved_candidates.size()) >= M) {
        break;
      }

      auto [distance_to_query, current_node_id] = candidates.top();
      distance_to_query = -distance_to_query;
      candidates.pop();

      bool should_keep_candidate = true;

      for (const auto &[saved_dist, saved_node_id] : saved_candidates) {
        float cur_dist = _index->_distance->distance(
            _index->getNodeData(saved_node_id),
            _index->getNodeData(current_node_id));

        float threshold = distance_to_query;

        // Apply strategy-specific threshold modification
        threshold = applyStrategyModification(threshold, current_node_id,
                                              saved_node_id, new_node_id);

        if (cur_dist < threshold) {
          should_keep_candidate = false;
          break;
        }
      }

      if (should_keep_candidate) {
        saved_candidates.push_back({-distance_to_query, current_node_id});
      }
    }

    // Reconstruct neighbors queue
    for (const dist_node_t &current_pair : saved_candidates) {
      neighbors.emplace(-current_pair.first, current_pair.second);
    }
  }

  /**
   * @brief Apply strategy-specific modification to pruning threshold.
   */
  float applyStrategyModification(float threshold, node_id_t candidate_id,
                                  node_id_t neighbor_id, node_id_t new_node_id) {
    switch (_config.strategy) {
    case TwoPassStrategy::HUBNESS_SCORING:
      return applyHubnessPenalty(threshold, candidate_id, neighbor_id);

    case TwoPassStrategy::EDGE_QUALITY_SCORING:
      return applyEdgeQualityBonus(threshold, candidate_id, new_node_id);

    default:
      return threshold;
    }
  }

  /**
   * @brief Apply hubness penalty: increase threshold for hub candidates.
   *
   * Nodes with high in-degree are "hubs" - we want to avoid connecting to them
   * too often to improve graph navigability. By increasing the threshold for
   * hub candidates, we make them easier to prune.
   */
  float applyHubnessPenalty(float threshold, node_id_t candidate_id,
                            node_id_t neighbor_id) {
    // Penalize if the CANDIDATE is a hub (high indegree from Pass 1)
    uint32_t candidate_indegree = _indegree_cache[candidate_id];
    // Max expected indegree based on Pass 1 edges
    float max_indegree =
        static_cast<float>(_config.M_pass1 * 2);

    float normalized_indegree =
        std::min(1.0f, static_cast<float>(candidate_indegree) / max_indegree);

    // Increase threshold for hub candidates = easier to prune them
    // This encourages connecting to non-hub nodes for better graph diversity
    float penalty = _config.hubness_penalty_weight * normalized_indegree;
    return threshold * (1.0f + penalty);
  }

  /**
   * @brief Apply edge quality bonus: prefer candidates with good quality.
   *
   * Edges with distance close to the node's average neighbor distance
   * are considered higher quality.
   */
  float applyEdgeQualityBonus(float threshold, node_id_t candidate_id,
                              node_id_t new_node_id) {
    float avg_dist = _node_avg_neighbor_dist[candidate_id];
    if (avg_dist <= 0.0f) {
      return threshold;
    }

    // Compute distance from new_node to candidate
    float dist = _index->_distance->distance(
        _index->getNodeData(new_node_id), _index->getNodeData(candidate_id));

    // Quality ratio: closer to 1.0 means distance is close to average
    // Values < 1 mean better than average (shorter distance)
    float quality_ratio = dist / avg_dist;

    // Bonus for high-quality edges: reduce threshold to favor keeping them
    if (quality_ratio < 1.0f) {
      // Good edge (shorter than average) - small bonus
      float bonus = (1.0f - quality_ratio) * _config.edge_quality_threshold;
      return threshold * (1.0f - bonus * 0.5f);
    } else {
      // Poor edge (longer than average) - small penalty
      float penalty =
          std::min(1.0f, quality_ratio - 1.0f) * _config.edge_quality_threshold;
      return threshold * (1.0f + penalty * 0.5f);
    }
  }
};

} // namespace flatnav
