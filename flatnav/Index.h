#pragma once
#include "DistanceInterface.h"
#include "util/ExplicitSet.h"
#include "util/reordering.h"
#include "util/verifysimd.h"

#include <algorithm> // for std::min
#include <cstring>
#include <fstream>
#include <limits>  // for std::numeric_limits<T>::max()
#include <queue>   // for std::priority_queue
#include <utility> // for std::pair
#include <vector>

namespace flatnav {

template <typename dist_t, typename label_t>
// dist_t: A distance function implementing DistanceInterface.
// label_t: A fixed-width data type for the label (meta-data) of each point.
class Index {
public:
  typedef std::pair<float, label_t> dist_label_t;

  Index(DistanceInterface<dist_t> dist, int num_data, int max_edges)
      : distance(std::move(dist)), max_num_nodes(num_data), cur_num_nodes(0),
        M(max_edges), is_visited(num_data + 1) {

    data_size_bytes = distance.data_size();
    node_size_bytes = data_size_bytes + sizeof(node_id_t) * M + sizeof(label_t);
    size_t index_memory_size = node_size_bytes * max_num_nodes;
    index_memory = new char[index_memory_size];
    transformed_query = new char[data_size_bytes];
  }

  Index(std::ofstream &in)
      : max_num_nodes(0), cur_num_nodes(0), index_memory(NULL),
        transformed_query(NULL) {
    deserialize(in);
  }

  ~Index() {
    delete[] index_memory;
    delete[] transformed_query;
  }

  bool add(void *data, label_t &label, int ef_construction,
           int num_initializations = 100) {
    // initialization must happen before alloc due to a bug where
    // searchInitialization chooses new_node_id as the initialization
    // since new_node_id has distance 0 (but no links). The search is
    // skipped because the "optimal" node seems to have been found.
    node_id_t new_node_id;
    node_id_t entry_node = searchInitialization(data, num_initializations);
    // make space for the new node
    if (!allocateNode(data, label, new_node_id)) {
      return false;
    }
    // search graph for neighbors of new node, connect to them
    if (new_node_id > 0) {
      PriorityQueue neighbors = beamSearch(data, entry_node, ef_construction);
      selectNeighbors(neighbors, M);
      connectNeighbors(neighbors, new_node_id);
    } else {
      return false;
    }
    return true;
  }

  std::vector<dist_label_t> search(const void *query, const int num_results,
                                   int ef_search,
                                   int num_initializations = 100) {

    // We use a pre-allocated buffer for the transformed query, for speed
    // reasons, but it would also be acceptable to manage this buffer
    // dynamically (e.g. in the multi-threaded setting).
    distance.transform_data(transformed_query, query);
    node_id_t entry_node =
        searchInitialization(transformed_query, num_initializations);
    PriorityQueue neighbors =
        beamSearch(transformed_query, entry_node, ef_search);
    std::vector<dist_label_t> results;
    while (neighbors.size() > num_results) {
      neighbors.pop();
    }
    while (neighbors.size() > 0) {
      results.emplace_back(neighbors.top().first,
                           *nodeLabel(neighbors.top().second));
      neighbors.pop();
    }
    std::sort(results.begin(), results.end(),
              [](const dist_label_t &left, const dist_label_t &right) {
                return left.first < right.first;
              });
    return results;
  }

  void serialize(std::ofstream &out) {
    // TODO: Make this safe across machines and compilers.
    distance.serialize(out);
    out.write(reinterpret_cast<char *>(&data_size_bytes), sizeof(size_t));
    out.write(reinterpret_cast<char *>(&node_size_bytes), sizeof(size_t));
    out.write(reinterpret_cast<char *>(&max_num_nodes), sizeof(size_t));
    out.write(reinterpret_cast<char *>(&cur_num_nodes), sizeof(size_t));
    out.write(reinterpret_cast<char *>(&M), sizeof(size_t));
    // Write the index data partition.
    size_t index_memory_size = node_size_bytes * max_num_nodes;
    out.write(reinterpret_cast<char *>(index_memory), index_memory_size);
  }

  void deserialize(std::ifstream &in) {
    distance.deserialize(in);
    in.read(reinterpret_cast<char *>(&data_size_bytes), sizeof(size_t));
    in.read(reinterpret_cast<char *>(&node_size_bytes), sizeof(size_t));
    in.read(reinterpret_cast<char *>(&max_num_nodes), sizeof(size_t));
    in.read(reinterpret_cast<char *>(&cur_num_nodes), sizeof(size_t));
    in.read(reinterpret_cast<char *>(&M), sizeof(size_t));

    if (data_size_bytes != distance.data_size()) {
      throw std::invalid_argument(
          "Error reading index: Data size from the index does not "
          "match the data size from the distance. Is the dimension "
          "correct?");
    }
    size_t node_size_bytes_check =
        data_size_bytes + sizeof(node_id_t) * M + sizeof(label_t);
    if (node_size_bytes != node_size_bytes_check) {
      throw std::invalid_argument(
          "Error reading index: The node size from the index does not "
          "match the expected node size based on max_edges, the vector "
          "size and the label type.");
    }

    if (index_memory != NULL) {
      delete[] index_memory;
      index_memory = NULL;
    }
    size_t index_memory_size = node_size_bytes * max_num_nodes;
    index_memory = new char[index_memory_size];
    in.read(reinterpret_cast<char *>(index_memory), index_memory_size);

    if (transformed_query != NULL) {
      delete[] transformed_query;
      transformed_query = NULL;
    }
    transformed_query = new char[data_size_bytes];

    is_visited = VisitedSet(max_num_nodes + 1);
  }

  void reorder_gorder(const int window_size = 5) {
    std::vector<std::vector<node_id_t>> outdegree_table(cur_num_nodes);
    for (node_id_t node = 0; node < cur_num_nodes; node++) {
      node_id_t *links = nodeLinks(node);
      for (int i = 0; i < M; i++) {
        if (links[i] != node) {
          outdegree_table[node].push_back(links[i]);
        }
      }
    }
    std::vector<node_id_t> P = g_order<node_id_t>(outdegree_table, window_size);

    relabel(P);
  }

  void reorder_rcm() {
    // TODO: Remove code duplication for outdegree_table.
    std::vector<std::vector<node_id_t>> outdegree_table(cur_num_nodes);
    for (node_id_t node = 0; node < cur_num_nodes; node++) {
      node_id_t *links = nodeLinks(node);
      for (int i = 0; i < M; i++) {
        if (links[i] != node) {
          outdegree_table[node].push_back(links[i]);
        }
      }
    }
    std::vector<node_id_t> P = rcm_order<node_id_t>(outdegree_table);
    relabel(P);
  }

private:
  // internal node numbering scheme
  typedef unsigned int node_id_t;
  typedef std::pair<float, node_id_t> dist_node_t;

  struct CompareNodes {
    constexpr bool operator()(dist_node_t const &a,
                              dist_node_t const &b) const noexcept {
      return a.first < b.first;
    }
  };

  typedef ExplicitSet VisitedSet;
  typedef std::priority_queue<dist_node_t, std::vector<dist_node_t>,
                              CompareNodes>
      PriorityQueue;

  // Large (several GB), pre-allocated block of memory.
  char *index_memory;
  char *transformed_query;

  size_t M;
  // size of one data point (does not support variable-size data, strings)
  size_t data_size_bytes;
  // Node consists of: ([data] [M links] [data label]). This layout was chosen
  // after benchmarking - it's slightly more cache-efficient than others.
  size_t node_size_bytes;
  size_t max_num_nodes; // Determines size of internal pre-allocated memory
  size_t cur_num_nodes;

  DistanceInterface<dist_t> distance;

  // Remembers which nodes we've visited, to avoid re-computing distances.
  // Might be a caching problem in beamSearch - needs to be profiled.
  VisitedSet is_visited;

  char *nodeData(const node_id_t &n) {
    char *location = index_memory + (n * node_size_bytes);
    return location;
  }

  node_id_t *nodeLinks(const node_id_t &n) {
    char *location = index_memory + (n * node_size_bytes) + data_size_bytes;
    return reinterpret_cast<node_id_t *>(location);
  }

  label_t *nodeLabel(const node_id_t &n) {
    char *location = index_memory + n * node_size_bytes + data_size_bytes +
                     M * sizeof(node_id_t);
    return reinterpret_cast<label_t *>(location);
  }

  bool allocateNode(void *data, label_t &label, node_id_t &new_node_id) {
    if (cur_num_nodes >= max_num_nodes) {
      return false;
    }
    new_node_id = cur_num_nodes;

    // Transforms and writes data into the index at the correct location.
    distance.transform_data(nodeData(cur_num_nodes), data);
    *(nodeLabel(cur_num_nodes)) = label;

    node_id_t *links = nodeLinks(cur_num_nodes);
    for (int i = 0; i < M; i++) {
      links[i] = cur_num_nodes;
    }

    cur_num_nodes++;
    return true;
  }

  inline void swapNodes(node_id_t a, node_id_t b, void *temp_data,
                        node_id_t *temp_links, label_t *temp_label) {

    // stash b in temp
    std::memcpy(temp_data, nodeData(b), data_size_bytes);
    std::memcpy(temp_links, nodeLinks(b), M * sizeof(node_id_t));
    std::memcpy(temp_label, nodeLabel(b), sizeof(label_t));

    // place node at a in b
    std::memcpy(nodeData(b), nodeData(a), data_size_bytes);
    std::memcpy(nodeLinks(b), nodeLinks(a), M * sizeof(node_id_t));
    std::memcpy(nodeLabel(b), nodeLabel(a), sizeof(label_t));

    // put node b in a
    std::memcpy(nodeData(a), temp_data, data_size_bytes);
    std::memcpy(nodeLinks(a), temp_links, M * sizeof(node_id_t));
    std::memcpy(nodeLabel(a), temp_label, sizeof(label_t));

    return;
  }

  PriorityQueue beamSearch(const void *query, const node_id_t entry_node,
                           const int buffer_size) {

    // The query pointer should contain transformed data.
    // returns an iterable list of node_id_t's, sorted by distance (ascending)
    PriorityQueue neighbors;  // W in the HNSW paper
    PriorityQueue candidates; // C in the HNSW paper

    is_visited.clear();
    float dist = distance.distance(query, nodeData(entry_node));
    float max_dist = dist;

    candidates.emplace(-dist, entry_node);
    neighbors.emplace(dist, entry_node);
    is_visited.insert(entry_node);

    while (!candidates.empty()) {
      // get nearest element from candidates
      dist_node_t d_node = candidates.top();
      if ((-d_node.first) > max_dist) {
        break;
      }
      candidates.pop();
      node_id_t *d_node_links = nodeLinks(d_node.second);
      for (int i = 0; i < M; i++) {
        if (!is_visited[d_node_links[i]]) {
          // If we haven't visited the node yet.
          is_visited.insert(d_node_links[i]);
          dist = distance.distance(query, nodeData(d_node_links[i]));
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

  void selectNeighbors(PriorityQueue &neighbors, const int M) {
    // selects neighbors from the PriorityQueue, according to HNSW heuristic
    if (neighbors.size() < M) {
      return;
    }

    PriorityQueue candidates;
    std::vector<dist_node_t> saved_candidates;
    while (neighbors.size() > 0) {
      candidates.emplace(-neighbors.top().first, neighbors.top().second);
      neighbors.pop();
    }

    while (candidates.size() > 0) {
      if (saved_candidates.size() >= M) {
        break;
      }
      dist_node_t current_pair = candidates.top();
      candidates.pop();

      bool should_keep_candidate = true;
      for (const dist_node_t &second_pair : saved_candidates) {
        float cur_dist = distance.distance(nodeData(second_pair.second),
                                           nodeData(current_pair.second));
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
    node_id_t *new_node_links = nodeLinks(new_node_id);
    int i = 0; // iterates through links for "new_node_id"

    while (neighbors.size() > 0) {
      node_id_t neighbor_node_id = neighbors.top().second;
      // add link to the current new node
      new_node_links[i] = neighbor_node_id;
      // now do the back-connections (a little tricky)
      node_id_t *neighbor_node_links = nodeLinks(neighbor_node_id);
      bool is_inserted = false;
      for (int j = 0; j < M; j++) {
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
        float max_dist = distance.distance(nodeData(new_node_id),
                                           nodeData(neighbor_node_id));
        PriorityQueue candidates;
        candidates.emplace(max_dist, new_node_id);
        for (int j = 0; j < M; j++) {
          if (neighbor_node_links[j] != neighbor_node_id) {
            candidates.emplace(
                distance.distance(nodeData(neighbor_node_id),
                                  nodeData(neighbor_node_links[j])),
                neighbor_node_links[j]);
          }
        }
        selectNeighbors(candidates, M);
        // connect the pruned set of candidates, including self-loops:
        int j = 0;
        while (candidates.size() > 0) { // candidates
          neighbor_node_links[j] = candidates.top().second;
          candidates.pop();
          j++;
        }
        while (j < M) { // self-loops (unused links)
          neighbor_node_links[j] = neighbor_node_id;
          j++;
        }
      }
      // loop increments:
      i++;
      if (i >= M) {
        i = M;
      }
      neighbors.pop();
    }
  }

  node_id_t searchInitialization(const void *query, int num_initializations) {
    // select entry_node from a set of random entry point options
    int step_size = cur_num_nodes / num_initializations;
    if (step_size <= 0) {
      step_size = 1;
    }

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    for (node_id_t node = 0; node < cur_num_nodes; node += step_size) {
      float dist = distance.distance(query, nodeData(node));
      if (dist < min_dist) {
        min_dist = dist;
        entry_node = node;
      }
    }
    return entry_node;
  }

  void relabel(const std::vector<node_id_t> &P) {
    // 1. Rewire all of the node connections
    for (node_id_t n = 0; n < cur_num_nodes; n++) {
      node_id_t *links = nodeLinks(n);
      for (int m = 0; m < M; m++) {
        links[m] = P[links[m]];
      }
    }

    // 2. Physically re-layout the nodes (in place)
    char *temp_data = new char[data_size_bytes];
    node_id_t *temp_links = new node_id_t[M];
    label_t *temp_label = new label_t;

    // In this context, is_visited stores which nodes have been relocated
    // (it would be equivalent to name this variable "is_relocated").
    is_visited.clear();

    for (node_id_t n = 0; n < cur_num_nodes; n++) {
      if (!is_visited[n]) {

        node_id_t src = n;
        node_id_t dest = P[src];

        // swap node at src with node at dest
        swapNodes(src, dest, temp_data, temp_links, temp_label);

        // mark src as having been relocated
        is_visited.insert(src);

        // recursively relocate the node from "dest"
        while (!is_visited[dest]) {
          // mark node as having been relocated
          is_visited.insert(dest);
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