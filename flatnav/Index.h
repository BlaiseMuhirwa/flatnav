#pragma once

#include <algorithm>
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cstring>
#include <flatnav/DistanceInterface.h>
#include <flatnav/util/ExplicitSet.h>
#include <flatnav/util/PreprocesorUtils.h>
#include <flatnav/util/Reordering.h>
#include <flatnav/util/SIMDDistanceSpecializations.h>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

namespace flatnav {

// dist_t: A distance function implementing DistanceInterface.
// label_t: A fixed-width data type for the label (meta-data) of each point.
template <typename dist_t, typename label_t> class Index {
public:
  typedef std::pair<float, label_t> dist_label_t;

  // Constructor for serialization with cereal. Do not use outside of
  // this class.
  Index() = default;

  /**
   * @brief Construct a new Index object for approximate near neighbor search
   *
   * @param dist                A distance metric for the specific index
   * distance. Options include l2(euclidean) and inner product.
   * @param dataset_size        The maximum number of vectors that can be
   * inserted in the index.
   * @param max_edges_per_node  The maximum number of links per node.
   */
  Index(std::shared_ptr<DistanceInterface<dist_t>> dist, int dataset_size,
        int max_edges_per_node)
      : _M(max_edges_per_node), _max_node_count(dataset_size),
        _cur_num_nodes(0), _distance(dist), _visited_nodes(dataset_size + 1) {

    _data_size_bytes = _distance->dataSize();
    _node_size_bytes =
        _data_size_bytes + (sizeof(node_id_t) * _M) + sizeof(label_t);

    size_t index_memory_size = _node_size_bytes * _max_node_count;

    _index_memory = new char[index_memory_size];
  }

  /**
   * @brief Construct a new Index object using a pre-computed outdegree table.
   * The outdegree table is extracted from a Matrix Market file.
   *
   * @param dist              A distance metric for the specific index
   * distance. Options include l2(euclidean) and inner product.
   * @param outdegree_table  A table of outdegrees for each node in the graph.
   * Each vector in the table contains the IDs of the nodes to which it is
   * connected.
   */

  Index(std::shared_ptr<DistanceInterface<dist_t>> dist,
        const std::string &mtx_filename)
      : _cur_num_nodes(0), _distance(dist) {
    auto mtx_graph =
        flatnav::util::loadGraphFromMatrixMarket(mtx_filename.c_str());
    _outdegree_table = std::move(mtx_graph.adjacency_list);
    _max_node_count = _outdegree_table.value().size();
    _M = mtx_graph.max_num_edges;

    _visited_nodes = VisitedSet(_max_node_count);

    _data_size_bytes = _distance->dataSize();
    _node_size_bytes =
        _data_size_bytes + (sizeof(node_id_t) * _M) + sizeof(label_t);
    size_t index_memory_size = _node_size_bytes * _max_node_count;
    _index_memory = new char[index_memory_size];
  }

  ~Index() { delete[] _index_memory; }

  void buildGraphLinks() {
    if (!_outdegree_table.has_value()) {
      throw std::runtime_error("Cannot build graph links without outdegree "
                               "table. Please construct index with outdegree "
                               "table.");
    }

    for (node_id_t node = 0; node < _outdegree_table.value().size(); node++) {
      node_id_t *links = getNodeLinks(node);
      for (int i = 0; i < _M; i++) {
        if (i >= _outdegree_table.value()[node].size()) {
          links[i] = node;
        } else {
          links[i] = _outdegree_table.value()[node][i];
        }
      }
    }
  }

  std::vector<std::vector<label_t>> getGraphOutdegreeTable() {
    std::vector<std::vector<label_t>> outdegree_table(_cur_num_nodes);
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      node_id_t *links = getNodeLinks(node);
      for (int i = 0; i < _M; i++) {
        if (links[i] != node) {
          outdegree_table[node].push_back(links[i]);
        }
      }
    }
    return outdegree_table;
  }

  /**
   * @brief Add a new vector to the index.
   *
   * @param data                 The vector to add.
   * @param label                The label (meta-data) of the vector.
   * @param ef_construction      ef parameter in the HNSW paper.
   * @param num_initializations  Parameter determining how to choose an entry
   * point.
   */
  void add(void *data, label_t &label, int ef_construction,
           int num_initializations = 100) {
    // initialization must happen before alloc due to a bug where
    // initializeSearch chooses new_node_id as the initialization
    // since new_node_id has distance 0 (but no links). The search is
    // skipped because the "optimal" node seems to have been found.
    node_id_t new_node_id;
    node_id_t entry_node = initializeSearch(data, num_initializations);

    // make space for the new node
    allocateNode(data, label, new_node_id);

    // search graph for neighbors of new node, connect to them
    if (new_node_id > 0) {
      PriorityQueue neighbors =
          beamSearch(/* query = */ data, /* entry_node = */ entry_node,
                     /* buffer_size = */ ef_construction);
      selectNeighbors(/* neighbors = */ neighbors);
      connectNeighbors(neighbors, new_node_id);
    }
  }

  /***
   * @brief Search the index for the k nearest neighbors of the query.
   * @param query The query vector.
   * @param K The number of nearest neighbors to return.
   * @param ef_search The search beam width.
   * @param num_initializations The number of random initializations to use.
   */
  std::vector<dist_label_t> search(const void *query, const int K,
                                   int ef_search,
                                   int num_initializations = 100) {

    node_id_t entry_node = initializeSearch(query, num_initializations);
    PriorityQueue neighbors = beamSearch(/* query = */ query,
                                         /* entry_node = */ entry_node,
                                         /* buffer_size = */ ef_search);
    std::vector<dist_label_t> results;

    while (neighbors.size() > K) {
      neighbors.pop();
    }

    auto size = neighbors.size();
    results.reserve(size);
    while (neighbors.size() > 0) {
      results.push_back(std::make_pair(neighbors.top().first,
                                       *getNodeLabel(neighbors.top().second)));
      neighbors.pop();
    }
    std::sort(results.begin(), results.end(),
              [](const dist_label_t &left, const dist_label_t &right) {
                return left.first < right.first;
              });
    return results;
  }

  // TODO: Add optional argument here for quantized data vector.
  void allocateNode(void *data, label_t &label, uint32_t &new_node_id) {
    if (_cur_num_nodes >= _max_node_count) {
      throw std::runtime_error("Maximum number of nodes reached. Consider "
                               "increasing the `max_node_count` parameter to "
                               "create a larger index.");
    }
    new_node_id = _cur_num_nodes;

    _distance->transformData(/* destination = */ getNodeData(new_node_id),
                             /* src = */ data);
    *(getNodeLabel(_cur_num_nodes)) = label;

    node_id_t *links = getNodeLinks(_cur_num_nodes);
    for (uint32_t i = 0; i < _M; i++) {
      links[i] = _cur_num_nodes;
    }

    _cur_num_nodes++;
  }

  void reorderGOrder(const int window_size = 5) {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P = gOrder<node_id_t>(outdegree_table, window_size);

    relabel(P);
  }

  void reorderRCM() {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P = rcmOrder<node_id_t>(outdegree_table);
    relabel(P);
  }

  static std::unique_ptr<Index<dist_t, label_t>>
  loadIndex(const std::string &filename) {
    std::ifstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    cereal::BinaryInputArchive archive(stream);
    std::unique_ptr<Index<dist_t, label_t>> index =
        std::make_unique<Index<dist_t, label_t>>();

    std::shared_ptr<DistanceInterface<dist_t>> dist =
        std::make_shared<dist_t>();

    // 1. Deserialize metadata
    archive(index->_M, index->_data_size_bytes, index->_node_size_bytes,
            index->_max_node_count, index->_cur_num_nodes, *dist,
            index->_visited_nodes);
    index->_distance = dist;

    // 3. Allocate memory using deserialized metadata
    index->_index_memory =
        new char[index->_node_size_bytes * index->_max_node_count];

    // 4. Deserialize content into allocated memory
    archive(
        cereal::binary_data(index->_index_memory,
                            index->_node_size_bytes * index->_max_node_count));

    return index;
  }

  void saveIndex(const std::string &filename) {
    std::ofstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    cereal::BinaryOutputArchive archive(stream);
    archive(*this);
  }

  inline size_t maxEdgesPerNode() const { return _M; }
  inline size_t dataSizeBytes() const { return _data_size_bytes; }

  inline size_t nodeSizeBytes() const { return _node_size_bytes; }

  inline size_t maxNodeCount() const { return _max_node_count; }

  inline char *indexMemory() const { return _index_memory; }
  inline size_t currentNumNodes() const { return _cur_num_nodes; }
  inline size_t dataDimension() const { return _distance->dimension(); }

  void printIndexParams() const {
    std::cout << "\nIndex Parameters" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    std::cout << "max_edges_per_node (M): " << _M << std::endl;
    std::cout << "data_size_bytes: " << _data_size_bytes << std::endl;
    std::cout << "node_size_bytes: " << _node_size_bytes << std::endl;
    std::cout << "max_node_count: " << _max_node_count << std::endl;
    std::cout << "cur_num_nodes: " << _cur_num_nodes << std::endl;
    std::cout << "visited_nodes size: " << _visited_nodes.size() << std::endl;

    _distance->printParams();
  }

private:
  // internal node numbering scheme. We might need to change this to uint64_t
  typedef uint32_t node_id_t;
  typedef std::pair<float, node_id_t> dist_node_t;

  typedef ExplicitSet VisitedSet;

  // NOTE: by default this is a max-heap. We could make this a min-heap
  // by using std::greater, but we want to use the queue as both a max-heap and
  // min-heap depending on the context.
  typedef std::priority_queue<dist_node_t, std::vector<dist_node_t>>
      PriorityQueue;

  // Large (several GB), pre-allocated block of memory.
  char *_index_memory;

  size_t _M;
  // size of one data point (does not support variable-size data, strings)
  size_t _data_size_bytes;
  // Node consists of: ([data] [M links] [data label]). This layout was chosen
  // after benchmarking - it's slightly more cache-efficient than others.
  size_t _node_size_bytes;
  size_t _max_node_count; // Determines size of internal pre-allocated memory
  size_t _cur_num_nodes;

  std::shared_ptr<DistanceInterface<dist_t>> _distance;

  // Remembers which nodes we've visited, to avoid re-computing distances.
  // Might be a caching problem in beamSearch - needs to be profiled.
  VisitedSet _visited_nodes;
  std::optional<std::vector<std::vector<uint32_t>>> _outdegree_table;

  friend class cereal::access;

  template <typename Archive> void serialize(Archive &archive) {
    archive(_M, _data_size_bytes, _node_size_bytes, _max_node_count,
            _cur_num_nodes, *_distance, _visited_nodes);

    // Serialize the allocated memory for the index & query.
    archive(
        cereal::binary_data(_index_memory, _node_size_bytes * _max_node_count));
  }

  char *getNodeData(const node_id_t &n) const {
    char *location = _index_memory + (n * _node_size_bytes);
    return location;
  }

  node_id_t *getNodeLinks(const node_id_t &n) const {
    char *location = _index_memory + (n * _node_size_bytes) + _data_size_bytes;
    return reinterpret_cast<node_id_t *>(location);
  }

  label_t *getNodeLabel(const node_id_t &n) const {
    char *location = _index_memory + (n * _node_size_bytes) + _data_size_bytes +
                     (_M * sizeof(node_id_t));
    return reinterpret_cast<label_t *>(location);
  }

  inline void swapNodes(node_id_t a, node_id_t b, void *temp_data,
                        node_id_t *temp_links, label_t *temp_label) {

    // stash b in temp
    std::memcpy(temp_data, getNodeData(b), _data_size_bytes);
    std::memcpy(temp_links, getNodeLinks(b), _M * sizeof(node_id_t));
    std::memcpy(temp_label, getNodeLabel(b), sizeof(label_t));

    // place node at a in b
    std::memcpy(getNodeData(b), getNodeData(a), _data_size_bytes);
    std::memcpy(getNodeLinks(b), getNodeLinks(a), _M * sizeof(node_id_t));
    std::memcpy(getNodeLabel(b), getNodeLabel(a), sizeof(label_t));

    // put node b in a
    std::memcpy(getNodeData(a), temp_data, _data_size_bytes);
    std::memcpy(getNodeLinks(a), temp_links, _M * sizeof(node_id_t));
    std::memcpy(getNodeLabel(a), temp_label, sizeof(label_t));

    return;
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

    // The query pointer should contain transformed data.
    // returns an iterable list of node_id_t's, sorted by distance (ascending)
    PriorityQueue neighbors;  // W in the HNSW paper
    PriorityQueue candidates; // C in the HNSW paper

    _visited_nodes.clear();
    float dist =
        _distance->distance(/* x = */ query, /* y = */ getNodeData(entry_node),
                            /* asymmetric = */ true);

    float max_dist = dist;

    candidates.emplace(-dist, entry_node);
    neighbors.emplace(dist, entry_node);
    _visited_nodes.insert(entry_node);

    while (!candidates.empty()) {
      // get nearest element from candidates
      dist_node_t d_node = candidates.top();
      if ((-d_node.first) > max_dist) {
        break;
      }
      candidates.pop();
      node_id_t *d_node_links = getNodeLinks(d_node.second);
      for (int i = 0; i < _M; i++) {
        if (!_visited_nodes[d_node_links[i]]) {
          // If we haven't visited the node yet.
          _visited_nodes.insert(d_node_links[i]);

          dist = _distance->distance(/* x = */ query,
                                     /* y = */ getNodeData(d_node_links[i]),
                                     /* asymmetric = */ true);

          // Include the node in the buffer if buffer isn't full or
          // if the node is closer than a node already in the buffer.
          if (neighbors.size() < buffer_size || dist < max_dist) {
            candidates.emplace(-dist, d_node_links[i]);
            neighbors.emplace(dist, d_node_links[i]);
            if (neighbors.size() > buffer_size) {
              neighbors.pop();
            }
            if (!neighbors.empty()) {
              max_dist = neighbors.top().first;
            }
          }
        }
      }
    }
    return neighbors;
  }

  void selectNeighbors(PriorityQueue &neighbors) {
    // selects neighbors from the PriorityQueue, according to the HNSW
    // heuristic
    if (neighbors.size() < _M) {
      return;
    }

    PriorityQueue candidates;
    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(_M);

    while (neighbors.size() > 0) {
      candidates.emplace(-neighbors.top().first, neighbors.top().second);
      neighbors.pop();
    }

    float cur_dist = 0.0;
    while (candidates.size() > 0) {
      if (saved_candidates.size() >= _M) {
        break;
      }
      // Extract the closest element from candidates.
      dist_node_t current_pair = candidates.top();
      candidates.pop();

      bool should_keep_candidate = true;
      for (const dist_node_t &second_pair : saved_candidates) {

        cur_dist =
            _distance->distance(/* x = */ getNodeData(second_pair.second),
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
    node_id_t *new_node_links = getNodeLinks(new_node_id);
    int i = 0; // iterates through links for "new_node_id"

    while (neighbors.size() > 0) {
      node_id_t neighbor_node_id = neighbors.top().second;
      // add link to the current new node
      new_node_links[i] = neighbor_node_id;
      // now do the back-connections (a little tricky)
      node_id_t *neighbor_node_links = getNodeLinks(neighbor_node_id);
      bool is_inserted = false;
      for (size_t j = 0; j < _M; j++) {
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
            _distance->distance(/* x = */ getNodeData(neighbor_node_id),
                                /* y = */ getNodeData(new_node_id));

        PriorityQueue candidates;
        candidates.emplace(max_dist, new_node_id);
        for (size_t j = 0; j < _M; j++) {
          if (neighbor_node_links[j] != neighbor_node_id) {
            auto label = neighbor_node_links[j];
            auto distance =
                _distance->distance(/* x = */ getNodeData(neighbor_node_id),
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
        while (j < _M) { // self-loops (unused links)
          neighbor_node_links[j] = neighbor_node_id;
          j++;
        }
      }
      // loop increments:
      i++;
      if (i >= _M) {
        i = _M;
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
    assert(num_initializations != 0);

    int step_size = _cur_num_nodes / num_initializations;
    if (step_size <= 0) {
      step_size = 1;
    }

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    for (node_id_t node = 0; node < _cur_num_nodes; node += step_size) {

      float dist =
          _distance->distance(/* x = */ query, /* y = */ getNodeData(node),
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
      for (int m = 0; m < _M; m++) {
        links[m] = P[links[m]];
      }
    }

    // 2. Physically re-layout the nodes (in place)
    char *temp_data = new char[_data_size_bytes];
    node_id_t *temp_links = new node_id_t[_M];
    label_t *temp_label = new label_t;

    // In this context, is_visited stores which nodes have been relocated
    // (it would be equivalent to name this variable "is_relocated").
    _visited_nodes.clear();

    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      if (!_visited_nodes[n]) {

        node_id_t src = n;
        node_id_t dest = P[src];

        // swap node at src with node at dest
        swapNodes(src, dest, temp_data, temp_links, temp_label);

        // mark src as having been relocated
        _visited_nodes.insert(src);

        // recursively relocate the node from "dest"
        while (!_visited_nodes[dest]) {
          // mark node as having been relocated
          _visited_nodes.insert(dest);
          // the value of src remains the same. However, dest needs
          // to change because the node located at src was previously
          // located at dest, and must be relocated to P[dest].
          dest = P[dest];

          // swap node at src with node at dest
          swapNodes(src, dest, temp_data, temp_links, temp_label);
        }
      }
    }

    delete[] temp_data;
    delete[] temp_links;
    delete temp_label;
  }
};

} // namespace flatnav