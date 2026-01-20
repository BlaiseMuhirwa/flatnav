#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/index/Index.h>
#include <flatnav/util/Multithreading.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

using flatnav::distances::DistanceInterface;

namespace flatnav {

/**
 * @brief Strategy variants for ensemble construction.
 */
enum class EnsembleVariant {
  /// Build k graphs with different insertion orders, accumulate candidates
  /// Each graph uses the previous graph's edges for routing (faster but less diverse)
  MULTI_ORDER_INCREMENTAL,

  /// Build k graphs with different insertion orders, reset edges between builds
  /// More diverse candidates but slower (no routing help between graphs)
  MULTI_ORDER_RESET,

  /// Build k graphs with different random seeds for entry point selection
  /// Lighter variation - same order, different starting points
  MULTI_SEED,
};

/**
 * @brief Configuration for ensemble construction.
 */
struct EnsembleConfig {
  /// Which ensemble variant to use
  EnsembleVariant variant = EnsembleVariant::MULTI_ORDER_INCREMENTAL;

  /// Number of graphs in the ensemble (typically 2-4)
  int num_graphs = 2;

  /// Edges per node for each ensemble graph (cheap graph)
  /// Final M = M_per_graph (we prune from num_graphs * M_per_graph candidates)
  int M_per_graph = 8;

  /// Final edges per node after pruning
  int M_final = 16;

  /// ef_construction for each ensemble graph (low for speed)
  int ef_construction_per_graph = 25;

  /// ef_construction for final pruning pass (higher for quality)
  int ef_construction_final = 100;

  /// Number of random entry points for search initialization
  int num_initializations = 100;

  /// Weight for hubness penalty during final pruning
  float hubness_penalty_weight = 0.1f;

  /// Number of threads for parallel construction
  uint32_t num_threads = 1;

  /// Random seed base (each graph uses seed_base + graph_index)
  uint64_t seed_base = 42;

  /// Whether to use neighbor expansion for candidate collection
  /// If false, uses the edges directly from each graph
  bool use_neighbor_expansion = true;

  /// Number of hops for neighbor expansion (2 or 3)
  int neighbor_expansion_hops = 2;
};

/**
 * @brief Ensemble-based index construction for improved graph quality.
 *
 * NEW FASTER APPROACH:
 * 1. Build ONE graph with baseline parameters (not k graphs!)
 * 2. Collect additional candidates via random beam searches
 * 3. Refine edges in-place using accumulated candidates
 *
 * This is much faster than the naive k-graph approach because:
 * - We only build 1 graph instead of k graphs
 * - Additional candidate collection is just beam search (no graph modification)
 * - Edge refinement is cheap (just update link arrays)
 *
 * @tparam dist_t Distance function type (e.g., SquaredL2Distance)
 * @tparam label_t Label type for node metadata
 */
template <typename dist_t, typename label_t>
class EnsembleBuilder {
public:
  using IndexType = Index<dist_t, label_t>;
  using node_id_t = uint32_t;
  using dist_node_t = std::pair<float, node_id_t>;
  using PriorityQueue = typename IndexType::PriorityQueue;

private:
  std::unique_ptr<IndexType> _index;
  EnsembleConfig _config;
  size_t _dataset_size;
  size_t _dimension;

  // Candidate accumulator: candidates[node_id] = vector of (distance, neighbor_id)
  std::vector<std::vector<dist_node_t>> _candidate_pool;

  // Per-node mutex for thread-safe candidate pool access
  std::unique_ptr<std::mutex[]> _candidate_pool_mutexes;

  // Hubness statistics computed from ensemble graphs
  std::unique_ptr<std::atomic<uint32_t>[]> _indegree_accumulated;

  // Insertion orders for each graph (precomputed)
  std::vector<std::vector<uint32_t>> _insertion_orders;

public:
  /**
   * @brief Construct an EnsembleBuilder.
   *
   * @param dist Distance function for the index
   * @param dataset_size Number of vectors to be indexed
   * @param config Ensemble construction configuration
   */
  EnsembleBuilder(std::unique_ptr<DistanceInterface<dist_t>> dist,
                  size_t dataset_size, const EnsembleConfig &config)
      : _config(config), _dataset_size(dataset_size),
        _dimension(dist->dimension()) {

    // Allocate index with final M edges
    _index = std::make_unique<IndexType>(
        std::move(dist), static_cast<int>(dataset_size), _config.M_final,
        /* collect_stats = */ false);

    _index->setNumThreads(_config.num_threads);

    // Pre-allocate candidate pool and mutexes
    _candidate_pool.resize(dataset_size);
    _candidate_pool_mutexes.reset(new std::mutex[dataset_size]);

    // Pre-allocate indegree tracking (initialized to 0)
    _indegree_accumulated.reset(new std::atomic<uint32_t>[dataset_size]());
    for (size_t i = 0; i < dataset_size; i++) {
      _indegree_accumulated[i].store(0, std::memory_order_relaxed);
    }

    // Generate random orderings for candidate collection passes
    generateInsertionOrders();
  }

  /**
   * @brief Build the index using ensemble construction.
   *
   * FAST APPROACH:
   * 1. Build one graph with baseline parameters
   * 2. Collect initial candidates from the built graph
   * 3. For each additional pass, collect more candidates via beam search
   * 4. Refine edges in-place using all accumulated candidates
   *
   * @tparam data_type Data type of the vectors (float, int8_t, uint8_t)
   * @param data Pointer to the vector data
   * @param labels Vector of labels for each data point
   */
  template <typename data_type>
  void build(void *data, std::vector<label_t> &labels) {
    // Step 1: Build initial graph using standard construction
    // This uses M_final edges and ef_construction_final
    buildInitialGraph<data_type>(data, labels);

    // Step 2: Collect candidates from the initial graph
    collectCandidatesFromGraph();

    // Step 3: For each additional pass, collect more candidates via beam search
    // using different random query orderings
    for (int pass = 1; pass < _config.num_graphs; pass++) {
      collectAdditionalCandidates<data_type>(data, pass);
    }

    // Step 4: Refine edges using accumulated candidates
    refineEdges<data_type>(data);
  }

  /**
   * @brief Get the built index.
   */
  IndexType &getIndex() { return *_index; }

  /**
   * @brief Release ownership of the built index.
   */
  std::unique_ptr<IndexType> releaseIndex() { return std::move(_index); }

private:
  /**
   * @brief Generate randomized orderings for candidate collection passes.
   */
  void generateInsertionOrders() {
    _insertion_orders.resize(_config.num_graphs);

    for (int g = 0; g < _config.num_graphs; g++) {
      _insertion_orders[g].resize(_dataset_size);
      std::iota(_insertion_orders[g].begin(), _insertion_orders[g].end(), 0);

      // Use different seed for each pass
      std::mt19937_64 rng(_config.seed_base + g);
      std::shuffle(_insertion_orders[g].begin(), _insertion_orders[g].end(), rng);
    }
  }

  /**
   * @brief Build the initial graph using standard HNSW construction.
   *
   * This builds a single graph with M_final edges and ef_construction_final,
   * similar to baseline but will be refined with additional candidates.
   */
  template <typename data_type>
  void buildInitialGraph(void *data, std::vector<label_t> &labels) {
    // Use standard index construction with full parameters
    _index->template addBatch<data_type>(
        data,
        labels,
        _config.ef_construction_final,
        _config.num_initializations
    );
  }

  /**
   * @brief Collect candidates from the current graph edges.
   *
   * This extracts all current edges as candidate neighbors.
   */
  void collectCandidatesFromGraph() {
    uint32_t num_nodes = _index->currentNumNodes();

    auto collectForNode = [&](uint32_t node_id) {
      node_id_t *links = _index->getNodeLinks(node_id);

      // Collect all current edges as candidates
      for (int j = 0; j < _config.M_final; j++) {
        node_id_t neighbor_id = links[j];
        if (neighbor_id != node_id && neighbor_id < num_nodes) {
          float dist = _index->_distance->distance(
              _index->getNodeData(node_id),
              _index->getNodeData(neighbor_id));

          {
            std::lock_guard<std::mutex> lock(_candidate_pool_mutexes[node_id]);
            _candidate_pool[node_id].emplace_back(dist, neighbor_id);
          }

          _indegree_accumulated[neighbor_id].fetch_add(1, std::memory_order_relaxed);
        }
      }

      // Also collect 2-hop neighbors for more diversity
      if (_config.use_neighbor_expansion) {
        collectNeighborExpansion(node_id, num_nodes);
      }
    };

    if (_config.num_threads > 1) {
      flatnav::executeInParallel(0, num_nodes, _config.num_threads, collectForNode);
    } else {
      for (uint32_t i = 0; i < num_nodes; i++) {
        collectForNode(i);
      }
    }
  }

  /**
   * @brief Collect 2-hop neighbors as additional candidates.
   */
  void collectNeighborExpansion(node_id_t node_id, uint32_t num_nodes) {
    auto *visited = _index->_visited_set_pool->pollAvailableSet();
    visited->clear();
    visited->insert(node_id);

    node_id_t *links = _index->getNodeLinks(node_id);
    std::vector<node_id_t> hop1_nodes;

    // Mark direct neighbors as visited
    for (int j = 0; j < _config.M_final; j++) {
      node_id_t neighbor = links[j];
      if (neighbor != node_id && neighbor < num_nodes) {
        visited->insert(neighbor);
        hop1_nodes.push_back(neighbor);
      }
    }

    // 2-hop: neighbors of neighbors
    for (node_id_t hop1 : hop1_nodes) {
      node_id_t *hop1_links = _index->getNodeLinks(hop1);
      for (int j = 0; j < _config.M_final; j++) {
        node_id_t candidate = hop1_links[j];
        if (candidate != hop1 &&
            candidate < num_nodes &&
            !visited->isVisited(candidate)) {
          visited->insert(candidate);

          float dist = _index->_distance->distance(
              _index->getNodeData(node_id),
              _index->getNodeData(candidate));

          {
            std::lock_guard<std::mutex> lock(_candidate_pool_mutexes[node_id]);
            _candidate_pool[node_id].emplace_back(dist, candidate);
          }
        }
      }
    }

    _index->_visited_set_pool->pushVisitedSet(visited);
  }

  /**
   * @brief Collect additional candidates via beam search.
   *
   * For each node, perform beam search from random entry points to find
   * additional neighbor candidates. This doesn't modify the graph, just
   * collects candidates.
   */
  template <typename data_type>
  void collectAdditionalCandidates(void *data, int pass) {
    uint32_t num_nodes = _index->currentNumNodes();
    uint32_t data_dimension = _index->_distance->dimension();
    const auto &order = _insertion_orders[pass];

    // Use different random seed for entry point selection
    std::mt19937_64 rng(_config.seed_base + pass * 1000);

    auto collectForNode = [&](uint32_t idx) {
      node_id_t node_id = order[idx];

      // Get the query vector for this node
      uint64_t offset = static_cast<uint64_t>(node_id) *
                        static_cast<uint64_t>(data_dimension);
      void *query = reinterpret_cast<data_type *>(data) + offset;

      // Initialize search from random entry points
      auto entry_node = _index->initializeSearch(query, _config.num_initializations);

      // Beam search to find candidates
      auto candidates = _index->beamSearch(query, entry_node,
                                           _config.ef_construction_per_graph);

      // Add candidates to pool
      while (candidates.size() > 0) {
        auto [dist, candidate_id] = candidates.top();
        candidates.pop();

        if (candidate_id != node_id) {
          std::lock_guard<std::mutex> lock(_candidate_pool_mutexes[node_id]);
          _candidate_pool[node_id].emplace_back(dist, candidate_id);
        }
      }
    };

    if (_config.num_threads > 1) {
      flatnav::executeInParallel(0, num_nodes, _config.num_threads, collectForNode);
    } else {
      for (uint32_t i = 0; i < num_nodes; i++) {
        collectForNode(i);
      }
    }
  }

  /**
   * @brief Refine edges using accumulated candidates.
   *
   * For each node, select the best M_final neighbors from all collected
   * candidates using HNSW-style pruning.
   */
  template <typename data_type>
  void refineEdges(void *data) {
    uint32_t num_nodes = _index->currentNumNodes();

    // First, deduplicate candidates
    deduplicateCandidates();

    // Then refine edges for each node
    auto refineNode = [&](uint32_t node_id) {
      auto &candidates = _candidate_pool[node_id];

      if (candidates.empty()) {
        return;
      }

      // Use HNSW-style neighbor selection (RNG pruning)
      PriorityQueue selected;
      for (const auto &[dist, neighbor_id] : candidates) {
        selected.emplace(dist, neighbor_id);
      }

      // Apply select neighbors to prune
      _index->selectNeighbors(selected, _config.M_final);

      // Write the new edges
      node_id_t *links = _index->getNodeLinks(node_id);
      int slot = 0;

      // Extract from priority queue (note: it's a max-heap, so we need to handle order)
      std::vector<dist_node_t> final_neighbors;
      while (selected.size() > 0 && slot < _config.M_final) {
        final_neighbors.push_back(selected.top());
        selected.pop();
        slot++;
      }

      // Write edges (closest first)
      std::sort(final_neighbors.begin(), final_neighbors.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });

      for (int i = 0; i < static_cast<int>(final_neighbors.size()); i++) {
        links[i] = final_neighbors[i].second;
      }

      // Fill remaining slots with self-loops
      for (int i = static_cast<int>(final_neighbors.size()); i < _config.M_final; i++) {
        links[i] = node_id;
      }
    };

    if (_config.num_threads > 1) {
      flatnav::executeInParallel(0, num_nodes, _config.num_threads, refineNode);
    } else {
      for (uint32_t i = 0; i < num_nodes; i++) {
        refineNode(i);
      }
    }

    // Ensure bidirectional connectivity
    ensureBidirectionalEdges(num_nodes);
  }

  /**
   * @brief Deduplicate candidates for each node.
   */
  void deduplicateCandidates() {
    auto dedupeNode = [&](uint32_t node_id) {
      auto &candidates = _candidate_pool[node_id];

      if (candidates.size() <= 1) return;

      // Sort by node_id for deduplication
      std::sort(candidates.begin(), candidates.end(),
                [](const dist_node_t &a, const dist_node_t &b) {
                  return a.second < b.second;
                });

      // Remove duplicates, keeping the one with smallest distance
      std::vector<dist_node_t> deduped;
      deduped.reserve(candidates.size());

      node_id_t prev_id = std::numeric_limits<node_id_t>::max();
      float prev_dist = std::numeric_limits<float>::max();

      for (const auto &[dist, id] : candidates) {
        if (id != prev_id) {
          if (prev_id != std::numeric_limits<node_id_t>::max()) {
            deduped.emplace_back(prev_dist, prev_id);
          }
          prev_id = id;
          prev_dist = dist;
        } else {
          prev_dist = std::min(prev_dist, dist);
        }
      }
      if (prev_id != std::numeric_limits<node_id_t>::max()) {
        deduped.emplace_back(prev_dist, prev_id);
      }

      candidates = std::move(deduped);

      // Sort by distance for pruning
      std::sort(candidates.begin(), candidates.end(),
                [](const dist_node_t &a, const dist_node_t &b) {
                  return a.first < b.first;
                });
    };

    if (_config.num_threads > 1) {
      flatnav::executeInParallel(0, static_cast<uint32_t>(_dataset_size),
                                 _config.num_threads, dedupeNode);
    } else {
      for (uint32_t i = 0; i < _dataset_size; i++) {
        dedupeNode(i);
      }
    }
  }

  /**
   * @brief Ensure bidirectional edges for better navigability.
   */
  void ensureBidirectionalEdges(uint32_t num_nodes) {
    for (uint32_t node_id = 0; node_id < num_nodes; node_id++) {
      node_id_t *links = _index->getNodeLinks(node_id);

      for (int j = 0; j < _config.M_final; j++) {
        node_id_t neighbor_id = links[j];
        if (neighbor_id != node_id && neighbor_id < num_nodes) {
          addBackConnection(node_id, neighbor_id);
        }
      }
    }
  }

  /**
   * @brief Add a back-connection from neighbor to node.
   */
  void addBackConnection(node_id_t node_id, node_id_t neighbor_id) {
    node_id_t *neighbor_links = _index->getNodeLinks(neighbor_id);

    // Check if already connected or find empty slot
    for (int j = 0; j < _config.M_final; j++) {
      if (neighbor_links[j] == node_id) {
        return; // Already connected
      }
      if (neighbor_links[j] == neighbor_id) {
        // Found empty slot
        neighbor_links[j] = node_id;
        return;
      }
    }

    // No empty slot - try to replace worst edge if this one is better
    float new_dist = _index->_distance->distance(
        _index->getNodeData(neighbor_id), _index->getNodeData(node_id));

    int worst_slot = -1;
    float worst_dist = new_dist;

    for (int j = 0; j < _config.M_final; j++) {
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
};

} // namespace flatnav
