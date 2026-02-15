#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/index/Index.h>
#include <flatnav/util/Multithreading.h>
#include <flatnav/util/PerfCounters.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>

using flatnav::distances::DistanceInterface;

namespace flatnav {

/**
 * @brief Configuration for anchor-based two-pass construction.
 *
 * Anchor construction builds a high-quality graph on a small subset ("anchors")
 * of the data in Pass 1, then uses this anchor graph as a navigation layer to
 * efficiently find entry points when inserting the remaining bulk data in Pass 2.
 *
 * This avoids the expensive random initializeSearch at scale, where probing
 * random nodes across a 100M+ node index causes cold memory accesses.
 * Instead, the anchor graph stays hot in cache and provides high-quality
 * entry points via a short beam search.
 */
struct AnchorConfig {
  /// Fraction of data to use as anchor nodes (e.g., 0.01 = 1%, 0.1 = 10%)
  float anchor_fraction = 0.01f;

  /// Maximum edges per node (same for all nodes, anchors and bulk)
  int M = 16;

  /// ef_construction for anchor graph (high value for quality)
  int anchor_ef_construction = 500;

  /// ef_construction for bulk node insertion
  int bulk_ef_construction = 80;

  /// Number of anchor probes for entry point finding.
  /// We uniformly sample this many anchors and pick the closest as the
  /// entry point for the bulk beam search.
  int num_anchor_probes = 10;

  /// Number of random entry points for anchor graph construction
  /// (used by initializeSearch during Pass 1)
  int num_initializations = 100;

  /// Number of threads for parallel construction
  uint32_t num_threads = 1;

  /// Random seed for anchor selection
  int64_t seed = 42;

  /// Whether to collect distance computation statistics
  bool collect_stats = false;
};

/**
 * @brief Anchor-based two-pass index construction for billion-scale datasets.
 *
 * This implements asymmetric pass construction where:
 *
 * Pass 1 (Anchor Pass): Build a high-quality graph on a small subset of
 * the data (the "anchors," typically 1-10%). Uses high ef_construction
 * for quality since the subset is small.
 *
 * Pass 2 (Bulk Pass): Insert the remaining 90-99% of points. Instead of
 * expensive random initializeSearch (which probes random nodes across
 * the entire dataset, causing cold memory accesses), use the anchor graph
 * to find a good starting node via a short beam search through the
 * well-connected anchor subgraph.
 *
 * Why it's faster:
 * - Random init in a 100M dataset: O(num_init) random memory accesses
 *   across ~50GB+ of data. Each access is likely a cache miss.
 * - Anchor search: O(anchor_search_ef) accesses through a compact, hot
 *   subgraph. Much better cache behavior and higher quality entry points.
 * - Better entry points → fewer beam search hops → fewer distance computations.
 *
 * Complexity: Transforms construction from O(N log N) with expensive constants
 * into O(N_anchor * C_expensive + N_bulk * C_cheap) where C_cheap << C_expensive.
 *
 * @tparam dist_t Distance function type (e.g., SquaredL2Distance)
 * @tparam label_t Label type for node metadata
 */
template <typename dist_t, typename label_t>
class AnchorBuilder {
public:
  using IndexType = Index<dist_t, label_t>;
  using node_id_t = uint32_t;

private:
  std::unique_ptr<IndexType> _index;
  AnchorConfig _config;
  size_t _dataset_size;
  size_t _num_anchors;

  // anchor_indices[i] = original data index for anchor i.
  // Anchor i gets node ID i in the index.
  std::vector<uint32_t> _anchor_indices;

  // bulk_indices[j] = original data index for bulk point j.
  // Bulk point j gets node ID (_num_anchors + j) in the index.
  std::vector<uint32_t> _bulk_indices;

public:
  /**
   * @brief Construct an AnchorBuilder.
   *
   * @param dist Distance function for the index
   * @param dataset_size Total number of vectors to be indexed
   * @param config Anchor construction configuration
   */
  AnchorBuilder(std::unique_ptr<DistanceInterface<dist_t>> dist,
                size_t dataset_size, const AnchorConfig &config)
      : _config(config), _dataset_size(dataset_size) {

    _num_anchors = std::max(
        static_cast<size_t>(1),
        static_cast<size_t>(dataset_size * config.anchor_fraction));

    // Don't exceed the dataset size
    _num_anchors = std::min(_num_anchors, dataset_size);

    _index = std::make_unique<IndexType>(
        std::move(dist), static_cast<int>(dataset_size), config.M,
        /* collect_stats = */ config.collect_stats);

    _index->setNumThreads(config.num_threads);

    selectAnchors();
  }

  /**
   * @brief Build the index using anchor-based two-pass construction.
   *
   * @tparam data_type Scalar data type of the vectors (float, uint8_t, int8_t)
   * @param data Pointer to the vector data array
   * @param labels Vector of labels for each data point
   */
  template <typename data_type>
  void build(void *data, std::vector<label_t> &labels) {
#ifdef FLATNAV_PERF_COUNTERS
    perf::globalStats().reset();
#endif

    // Pass 1: Insert anchor nodes with high-quality parameters
    insertAnchors<data_type>(data, labels);

    // Pass 2: Insert remaining nodes using anchor graph for entry points
    insertBulk<data_type>(data, labels);
  }

  /**
   * @brief Get a reference to the built index.
   */
  IndexType &getIndex() { return *_index; }

  /**
   * @brief Release ownership of the built index.
   */
  std::unique_ptr<IndexType> releaseIndex() { return std::move(_index); }

  /**
   * @brief Get the number of anchor nodes.
   */
  size_t numAnchors() const { return _num_anchors; }

private:
  /**
   * @brief Select which data points serve as anchors.
   *
   * Randomly samples anchor_fraction of the data indices using a
   * Fisher-Yates partial shuffle with srand/rand (fast, no MT overhead).
   * Anchor indices are sorted for better memory access locality.
   */
  void selectAnchors() {
    std::vector<uint32_t> all_indices(_dataset_size);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    // Fisher-Yates partial shuffle: only need to shuffle the first
    // _num_anchors elements, not the entire array.
    srand(static_cast<unsigned>(_config.seed));
    for (size_t i = 0; i < _num_anchors; i++) {
      size_t j = i + static_cast<size_t>(rand()) % (_dataset_size - i);
      std::swap(all_indices[i], all_indices[j]);
    }

    _anchor_indices.assign(all_indices.begin(),
                           all_indices.begin() + _num_anchors);
    _bulk_indices.assign(all_indices.begin() + _num_anchors,
                         all_indices.end());

    // Sort anchor indices for data access locality
    std::sort(_anchor_indices.begin(), _anchor_indices.end());
  }

  /**
   * @brief Pass 1: Insert anchor nodes with high ef_construction.
   *
   * Anchors are inserted using the standard add() method, which handles
   * parallel construction correctly. They get node IDs 0 to _num_anchors-1.
   * High ef_construction ensures the anchor graph is well-connected,
   * making it an effective navigation layer for Pass 2.
   */
  template <typename data_type>
  void insertAnchors(void *data, std::vector<label_t> &labels) {
    uint32_t data_dimension = _index->_distance->dimension();

    auto insertAnchor = [&](uint32_t i) {
#ifdef FLATNAV_PERF_COUNTERS
      thread_local bool initialized = false;
      if (!initialized) {
        perf::initThreadCounters();
        initialized = true;
      }
#endif
      uint32_t data_idx = _anchor_indices[i];
      uint64_t offset = static_cast<uint64_t>(data_idx) *
                        static_cast<uint64_t>(data_dimension);
      void *vector = reinterpret_cast<data_type *>(data) + offset;
      label_t label = labels[data_idx];

      _index->add(vector, label, _config.anchor_ef_construction,
                  _config.num_initializations);
    };

    if (_config.num_threads == 1) {
      for (uint32_t i = 0; i < _num_anchors; i++) {
        insertAnchor(i);
      }
    } else {
      flatnav::executeInParallel(0, static_cast<uint32_t>(_num_anchors),
                                 _config.num_threads, insertAnchor);
    }
  }

  /**
   * @brief Pass 2: Insert bulk nodes using anchor graph for entry points.
   *
   * For each non-anchor vector, instead of random initializeSearch (which
   * probes random nodes across the entire dataset), we find a good entry
   * point by searching the anchor graph. This is faster because:
   * 1. Anchor nodes are compact in memory → better cache behavior
   * 2. The anchor graph is well-connected → short beam search finds
   *    high-quality entry points
   * 3. Better entry points → fewer beam search hops during insertion
   */
  template <typename data_type>
  void insertBulk(void *data, std::vector<label_t> &labels) {
    uint32_t data_dimension = _index->_distance->dimension();
    uint32_t num_bulk = static_cast<uint32_t>(_bulk_indices.size());

    auto insertBulkPoint = [&](uint32_t bulk_i) {
#ifdef FLATNAV_PERF_COUNTERS
      // Initialize perf counters for this thread on first call
      thread_local bool initialized = false;
      if (!initialized) {
        perf::initThreadCounters();
        initialized = true;
      }
#endif

      uint32_t data_idx = _bulk_indices[bulk_i];
      uint64_t offset = static_cast<uint64_t>(data_idx) *
                        static_cast<uint64_t>(data_dimension);
      void *vector = reinterpret_cast<data_type *>(data) + offset;
      label_t label = labels[data_idx];

      addBulkNode(vector, label);
    };

    if (_config.num_threads == 1) {
      for (uint32_t i = 0; i < num_bulk; i++) {
        insertBulkPoint(i);
      }
    } else {
      flatnav::executeInParallel(0, num_bulk, _config.num_threads,
                                 insertBulkPoint);
    }
  }

  /**
   * @brief Insert a single bulk node using the anchor graph for entry.
   *
   * This replaces the standard add() method's initializeSearch with
   * findEntryViaAnchors, which provides better entry points at scale.
   * The rest of the insertion (beam search, neighbor selection, connection)
   * is identical to standard HNSW construction.
   */
  void addBulkNode(void *data, label_t &label) {
    if (_index->_cur_num_nodes >= _index->_max_node_count) {
      throw std::runtime_error("Maximum number of nodes reached.");
    }

    node_id_t entry_node = findEntryViaAnchors(data);
    std::unique_lock<std::mutex> global_lock(_index->_index_data_guard);
    node_id_t new_node_id;
    _index->allocateNode(data, label, new_node_id);
    global_lock.unlock();

    if (new_node_id == 0) {
      return;
    }

    auto neighbors = _index->beamSearch(data, entry_node,
                                        _config.bulk_ef_construction);

    int selection_M = std::max(static_cast<int>(_index->_M / 2), 1);
    _index->selectNeighbors(neighbors, selection_M);
    _index->connectNeighbors(neighbors, new_node_id);
  }



  /**
   * @brief Find a good entry point by sampling anchor nodes.
   *
   * Same logic as initializeSearch but restricted to the first _num_anchors
   * nodes. At large scale this is faster than probing the full index because
   * anchor nodes are contiguous in memory (good cache locality) rather than
   * spread across hundreds of GB.
   *
   * @param query The query vector
   * @return node_id_t The closest sampled anchor node
   */
  node_id_t findEntryViaAnchors(const void *query) {
#ifdef FLATNAV_PERF_COUNTERS
    perf::ScopedCounter _perf_counter("findEntryViaAnchors");
#endif

    int num_probes = _config.num_anchor_probes;
    int step_size = std::max(
        1, static_cast<int>(_num_anchors / num_probes));

    float min_dist = std::numeric_limits<float>::max();
    node_id_t best_anchor = 0;

    for (node_id_t node = 0; node < _num_anchors; node += step_size) {
      float dist = _index->_distance->distance(
          query, _index->getNodeData(node), /* asymmetric = */ true);
      if (dist < min_dist) {
        min_dist = dist;
        best_anchor = node;
      }
    }

    if (_index->_collect_stats) {
      _index->_distance_computations.fetch_add(
          static_cast<uint64_t>((_num_anchors + step_size - 1) / step_size));
    }

    return best_anchor;
  }
};

} // namespace flatnav
