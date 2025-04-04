#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/index/Allocator.h>
#include <flatnav/index/PrimitiveTypes.h>
#include <optional>
#include <random>

namespace flatnav {

using flatnav::distances::DistanceInterface;

// Proposed Reduced Enum
enum class PruningHeuristic {
  ARYA_MOUNT,
  VAMANA,
  ARYA_MOUNT_SANITY_CHECK,
  NEAREST_M,
  FURTHEST_M,
  MEDIAN_ADAPTIVE,
  TOP_M_MEDIAN_ADAPTIVE,
  MEAN_SORTED_BASELINE,
  QUANTILE_NOT_MIN,  // Parameter: Quantile value (e.g., 0.2)
  ARYA_MOUNT_REVERSED,
  PROBABILISTIC_RANK,    // Parameter: Rank scale (e.g., 1.0)
  NEIGHBORHOOD_OVERLAP,  // Parameter: Overlap threshold (e.g., 0.8)
  GEOMETRIC_MEAN,
  SIGMOID_RATIO,  // Parameter: Steepness (e.g., 1.0, 5.0, 10.0)
  ARYA_MOUNT_SHUFFLED,
  ARYA_MOUNT_RANDOM_ON_REJECTS,   // Parameter: Probability (e.g., 0.01, 0.05, 0.1)
  ARYA_MOUNT_SIGMOID_ON_REJECTS,  // Parameter: Steepness (e.g., 0.1, 5.0, 10.0)
  CHEAP_OUTDEGREE_CONDITIONAL,  // Parameter: Threshold (e.g., 2, 4, ..., or special values for M/default)
  LARGE_OUTDEGREE_CONDITIONAL,  // Parameter: Threshold (uses local calculation for now)
  ONE_SPANNER,
  ARYA_MOUNT_PLUS_SPANNER
};

template <typename dist_t, typename label_t>
struct PruningHeuristicSelector {
  using dist_label_t = flatnav::index_dist_label_t<label_t>;
  using dist_node_t = flatnav::index_dist_node_t;
  using PriorityQueue = flatnav::index_priority_queue_t;
  using node_id_t = flatnav::index_node_id_t;
  using MemoryAllocator = flatnav::FlatMemoryAllocator<int>;

  // TODO: Make these references const
  MemoryAllocator& _allocator;
  DistanceInterface<dist_t>& _distance;

  PruningHeuristicSelector(MemoryAllocator& allocator,
                           DistanceInterface<dist_t>& distance)
      : _allocator(allocator), _distance(distance) {}

  void select(PruningHeuristic heuristic, PriorityQueue& neighbors, int M,
              std::optional<float> parameter = std::nullopt) const {
    switch (heuristic) {
      case PruningHeuristic::ARYA_MOUNT: {
        selectNeighborsAryaMount(neighbors, M);
        break;
      }
      case PruningHeuristic::ARYA_MOUNT_SANITY_CHECK:
        selectNeighborsAryaMountSanityCheck(neighbors, M);
        break;
      case PruningHeuristic::ARYA_MOUNT_SIGMOID_ON_REJECTS: {
        // sigmoid slope = 0.1.
        auto steepness = parameter.value_or(0.1f);
        selectNeighborsAryaMountSigmoidOnRejects(neighbors, M, steepness);
        break;
      }
      case PruningHeuristic::ARYA_MOUNT_RANDOM_ON_REJECTS: {
        auto probability = parameter.value_or(0.01f);
        selectNeighborsAryaMountRandomOnRejects(neighbors, M, probability);
        break;
      }
      case PruningHeuristic::ARYA_MOUNT_REVERSED:
        selectNeighborsAryaMountReversed(neighbors, M);
        break;
      case PruningHeuristic::ARYA_MOUNT_SHUFFLED:
        selectNeighborsAryaMountShuffled(neighbors, M);
        break;
      case PruningHeuristic::ARYA_MOUNT_PLUS_SPANNER:
        selectNeighborsAryaMountPlusSpanner(neighbors, M);
        break;
      case PruningHeuristic::VAMANA: {
        auto alpha = parameter.value_or(0.8333f);
        selectNeighborsVamana(neighbors, M, alpha);
        break;
      }
      case PruningHeuristic::NEAREST_M:
        selectNeighborsPickTopM(neighbors, M);
        break;
      case PruningHeuristic::FURTHEST_M:
        selectNeighborsPickTopM(neighbors, M, /* nearest = */ false);
        break;
      case PruningHeuristic::CHEAP_OUTDEGREE_CONDITIONAL: {
        int cheap_outgoing_edge_threshold = static_cast<int>(parameter.value_or(M));
        selectNeighborsCheapOutDegreeConditional(neighbors, M,
                                                 cheap_outgoing_edge_threshold);
        break;
      }
      case PruningHeuristic::LARGE_OUTDEGREE_CONDITIONAL: {
        int well_connected_outgoing_edge_threshold =
            static_cast<int>(parameter.value_or(M));
        selectNeighborsLargeOutDegreeConditional(neighbors, M,
                                                 well_connected_outgoing_edge_threshold);
        break;
      }
      case PruningHeuristic::SIGMOID_RATIO: {
        auto sigmoid_steepness = parameter.value_or(1.0f);
        selectNeighborsSigmoidRatio(neighbors, M, sigmoid_steepness);
        break;
      }
      case PruningHeuristic::MEDIAN_ADAPTIVE:
        selectNeighborsMedianAdaptive(neighbors, M);
        break;
      case PruningHeuristic::TOP_M_MEDIAN_ADAPTIVE:
        selectNeighborsTopMMeanAdaptive(neighbors, M);
        break;
      case PruningHeuristic::MEAN_SORTED_BASELINE:
        selectNeighborsMeanSortedBaseline(neighbors, M);
        break;
      case PruningHeuristic::QUANTILE_NOT_MIN: {
        auto quantile = parameter.value_or(0.2f);
        selectNeighborsQuantileNotMin(neighbors, M, quantile);
        break;
      }
      case PruningHeuristic::PROBABILISTIC_RANK: {
        auto rank_prune_factor = parameter.value_or(1.0f);
        selectNeighborsProbabilisticRank(neighbors, M, rank_prune_factor);
        break;
      }
      case PruningHeuristic::NEIGHBORHOOD_OVERLAP: {
        auto overlap_threshold = parameter.value_or(0.8f);
        selectNeighborsNeighborhoodOverlap(neighbors, M, overlap_threshold);
        break;
      }
      case PruningHeuristic::GEOMETRIC_MEAN:
        selectNeighborsGeometricMean(neighbors, M);
        break;
      case PruningHeuristic::ONE_SPANNER:
        selectNeighborsOneSpanner(neighbors, M);
        break;
      default:
        selectNeighborsAryaMount(neighbors, M);
    }
  }

  /**
   * @brief Selects neighbors from the PriorityQueue, according to the original
   * HNSW heuristic from Arya&Mount. The neighbors priority queue contains
   * elements sorted by distance where the top element is the furthest neighbor
   * from the query.
   */
  void selectNeighborsAryaMount(PriorityQueue& neighbors, int M) const {
    if (neighbors.size() < M) {
      return;
    }

    std::priority_queue<std::pair<float, node_id_t>> candidates;
    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);

    while (neighbors.size() > 0) {
      auto [distance, id] = neighbors.top();

      candidates.emplace(-distance, id);
      neighbors.pop();
    }

    while (candidates.size() > 0) {
      if (saved_candidates.size() >= M) {
        break;
      }
      // Extract the closest element from candidates.
      auto [distance_to_query, current_node_id] = candidates.top();
      distance_to_query = -distance_to_query;
      candidates.pop();

      bool should_keep_candidate = true;
      for (const auto& [_, second_pair_node_id] : saved_candidates) {
        float cur_dist = _distance.distance(
            /* x = */ _allocator.getNodeData(second_pair_node_id),
            /* y = */ _allocator.getNodeData(current_node_id));

        if (cur_dist < distance_to_query) {
          should_keep_candidate = false;
          break;
        }
      }
      if (should_keep_candidate) {
        // We could do neighbors.emplace except we have to iterate
        // through saved_candidates, and std::priority_queue doesn't
        // support iteration (there is no technical reason why not).
        auto current_pair = std::make_pair(-distance_to_query, current_node_id);
        saved_candidates.push_back(current_pair);
      }
    }
    // TODO: implement my own priority queue, get rid of vector
    // saved_candidates, add directly to neighborqueue earlier.
    for (const dist_node_t& current_pair : saved_candidates) {
      neighbors.emplace(-current_pair.first, current_pair.second);
    }
  }

  void selectNeighborsAryaMountSanityCheck(PriorityQueue& neighbors, int M) const {
    if (neighbors.size() < M) {
      return;
    }
    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order
              });

    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;  // Already the distance to the query
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);
      if (closest_saved_candidate_dist >= baseline_distance) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, (int)saved_candidates.size());
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsVamana(PriorityQueue& neighbors, int M, float alpha) const {
    if (neighbors.size() < M)
      return;

    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }

    std::sort(
        all_candidates.begin(), all_candidates.end(),
        [](const dist_node_t& a, const dist_node_t& b) { return a.first < b.first; });

    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);
      if (alpha * closest_saved_candidate_dist >= baseline_distance) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, static_cast<int>(saved_candidates.size()));
    for (int i = 0; i < loop_limit; ++i) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsPickTopM(PriorityQueue& neighbors, int M,
                               bool nearest = true) const {
    if (neighbors.size() < M) {
      return;
    }
    //Since 'neighbors' is already a priority queue sorted by distance,
    // we just need to keep the top M elements.
    int count = 0;
    std::vector<dist_node_t> all_candidates;  //Store all and sort
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    if (nearest) {
      std::sort(
          all_candidates.begin(), all_candidates.end(),
          [](const dist_node_t& a, const dist_node_t& b) { return a.first < b.first; });
    } else {
      std::sort(
          all_candidates.begin(), all_candidates.end(),
          [](const dist_node_t& a, const dist_node_t& b) { return a.first > b.first; });
    }

    for (const auto& candidate : all_candidates) {  //iterate through the candidates
      if (count < M) {
        neighbors.emplace(candidate.first, candidate.second);  //re-add to the queue
      }
      count++;
      if (count >= M)
        break;
    }
  }

  void selectNeighborsCheapOutDegreeConditional(PriorityQueue& neighbors, int M,
                                                int cheap_outgoing_edge_threshold) const {
    // if a node has fewer than "cheap_outgoing_edge_threshold" outbound links,
    // it's cheap to visit so we can include it at minimal cost even if Arya&Mount
    // would've pruned it.
    if (neighbors.size() < M) {
      return;
    }

    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (neighbors.size() > 0) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order.
              });
    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    std::vector<dist_node_t> saved_candidates;
    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;  // Already the distance to the query
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);

      node_id_t* candidate_links = _allocator.getNodeLinks(candidate.second);
      int candidate_outdegree = 0;
      for (size_t i = 0; i < M; ++i) {
        if (candidate_links[i] != candidate.second) {
          candidate_outdegree++;
        }
      }
      bool candidate_is_cheap = (candidate_outdegree <= cheap_outgoing_edge_threshold);
      if ((closest_saved_candidate_dist >= baseline_distance) || (candidate_is_cheap)) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit =
        std::min(M, static_cast<int>(saved_candidates.size()));  // Use effective_M!
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsLargeOutDegreeConditional(
      PriorityQueue& neighbors, int M, int well_connected_outgoing_edge_threshold) const {
    // if a node has more than "well_connected_outgoing_edge_threshold" outbound links,
    // it's expensive to visit but maybe worth it beacuse it has so many out-degree nodes.
    if (neighbors.size() < M) {
      return;
    }

    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (neighbors.size() > 0) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order.
              });
    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    std::vector<dist_node_t> saved_candidates;
    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;  // Already the distance to the query
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);

      node_id_t* candidate_links = _allocator.getNodeLinks(candidate.second);
      int candidate_outdegree = 0;
      for (size_t i = 0; i < M; ++i) {
        if (candidate_links[i] != candidate.second) {
          candidate_outdegree++;
        }
      }
      bool candidate_is_well_connected =
          (candidate_outdegree >= well_connected_outgoing_edge_threshold);
      if ((closest_saved_candidate_dist >= baseline_distance) ||
          (candidate_is_well_connected)) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit =
        std::min(M, static_cast<int>(saved_candidates.size()));  // Use effective_M!
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsSigmoidRatio(PriorityQueue& neighbors, int M,
                                   float steepness) const {
    if (neighbors.size() < M) {
      return;
    }
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (neighbors.size() > 0) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }

    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order.
              });
    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    std::vector<dist_node_t> saved_candidates;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);
      float ratio = (baseline_distance != 0.0f)
                        ? (closest_saved_candidate_dist / baseline_distance)
                        : 0.0f;

      // Sigmoid function for smooth thresholding
      float midpoint = 1.0;  // Place "meets threshold exactly" at 50% probability.
      float prune_probability =
          1.0f / (1.0f + std::exp(-1 * steepness * (ratio - midpoint)));

      // Generate a random number between 0 and 1
      float random_number = distrib(gen);

      if (random_number < prune_probability) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, static_cast<int>(saved_candidates.size()));
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsAryaMountSigmoidOnRejects(PriorityQueue& neighbors, int M,
                                                float steepness) const {
    if (neighbors.size() < M) {
      return;
    }
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (neighbors.size() > 0) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }

    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order.
              });
    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    std::vector<dist_node_t> saved_candidates;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);
      float ratio = (baseline_distance != 0.0f)
                        ? (closest_saved_candidate_dist / baseline_distance)
                        : 0.0f;

      // Sigmoid function for smooth thresholding
      float midpoint = 1.0;  // Place "meets threshold exactly" at 50% probability.
      float prune_probability =
          1.0f / (1.0f + std::exp(-1 * steepness * (ratio - midpoint)));

      // Generate a random number between 0 and 1
      float random_number = distrib(gen);

      if ((closest_saved_candidate_dist >= baseline_distance) ||
          (random_number < prune_probability)) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, static_cast<int>(saved_candidates.size()));
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsAryaMountRandomOnRejects(PriorityQueue& neighbors, int M,
                                               float accept_anyway_prob) const {
    if (neighbors.size() < M) {
      return;
    }

    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order
              });

    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;  // Already the distance to the query
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);

      // Generate a random number between 0 and 1
      float random_number = distrib(gen);

      bool should_accept_anyway = random_number <= accept_anyway_prob;
      if ((closest_saved_candidate_dist >= baseline_distance) || should_accept_anyway) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, (int)saved_candidates.size());
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsMedianAdaptive(PriorityQueue& neighbors, int M) const {
    if (neighbors.size() < M) {
      return;
    }

    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order
              });

    auto median_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                       node_id_t node_id) {
      if (saved.empty())
        return std::numeric_limits<float>::max();
      std::vector<float> distances;
      distances.reserve(saved.size());
      for (const auto& saved_node : saved) {
        distances.push_back(_distance.distance(_allocator.getNodeData(saved_node.second),
                                               _allocator.getNodeData(node_id)));
      }
      std::sort(distances.begin(), distances.end());
      size_t n = distances.size();
      return n % 2 == 0 ? (distances[n / 2 - 1] + distances[n / 2]) / 2.0f
                        : distances[n / 2];
    };

    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    for (const auto& candidate : all_candidates) {
      float baseline_distance =
          median_distance_to_node(saved_candidates, candidate.second);
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);
      if (closest_saved_candidate_dist >= baseline_distance) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, (int)saved_candidates.size());
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsTopMMeanAdaptive(PriorityQueue& neighbors, int M) const {
    if (neighbors.size() < M) {
      return;
    }

    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(
        all_candidates.begin(), all_candidates.end(),
        [](const dist_node_t& a, const dist_node_t& b) { return a.first < b.first; });
    //No need for take_first_M. We just take from the sorted candidates, with a limit

    auto mean_distance_to_node = [&](const std::vector<dist_node_t>& top_m,
                                     node_id_t node_id) {
      if (top_m.empty())
        return std::numeric_limits<float>::max();
      float sum_dist = 0.0f;

      for (const auto& top_node : top_m) {
        sum_dist += _distance.distance(_allocator.getNodeData(top_node.second),
                                       _allocator.getNodeData(node_id));
      }
      return sum_dist / top_m.size();
    };
    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    int top_m_count = 0;
    std::vector<dist_node_t> top_m_candidates;

    for (const auto& candidate : all_candidates) {  //get the top M candidates.
      if (top_m_count < M) {
        top_m_candidates.push_back(candidate);
      } else {
        break;
      }
      top_m_count++;
    }

    for (const auto& candidate : all_candidates) {
      float baseline_distance = mean_distance_to_node(top_m_candidates, candidate.second);
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);
      if (closest_saved_candidate_dist >= baseline_distance) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, (int)saved_candidates.size());
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsMeanSortedBaseline(PriorityQueue& neighbors, int M) const {
    if (neighbors.size() < M) {
      return;
    }

    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order
              });

    auto mean_distance_to_node = [&](const std::vector<dist_node_t>& candidates,
                                     node_id_t node_id) {
      if (candidates.empty())
        return std::numeric_limits<float>::max();
      float sum_dist = 0.0f;
      for (const auto& cand : candidates) {
        sum_dist += _distance.distance(_allocator.getNodeData(cand.second),
                                       _allocator.getNodeData(node_id));
      }
      return sum_dist / candidates.size();
    };

    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    for (const auto& candidate : all_candidates) {
      float baseline_distance =
          mean_distance_to_node(all_candidates, candidate.second);  // Use ALL candidates
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);
      if (closest_saved_candidate_dist >= baseline_distance) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, (int)saved_candidates.size());
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsQuantileNotMin(PriorityQueue& neighbors, int M,
                                     double quantile) const {
    if (neighbors.size() < M) {
      return;
    }

    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order
              });

    auto quantile_of = [&](const std::vector<float>& distances,
                           double quantile) -> float {
      if (distances.empty()) {
        return std::numeric_limits<float>::max();
      }
      // Create a copy to avoid modifying the original vector
      std::vector<float> sorted_distances = distances;
      std::sort(sorted_distances.begin(), sorted_distances.end());
      int index = static_cast<int>(std::ceil(
          quantile * (sorted_distances.size() - 1)));  // Corrected index calculation
      return sorted_distances[index];
    };

    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;
      std::vector<float> distances_to_candidate;
      //Get distances to candidate
      for (const auto& saved_node : saved_candidates) {
        distances_to_candidate.push_back(
            _distance.distance(_allocator.getNodeData(saved_node.second),
                               _allocator.getNodeData(candidate.second)));
      }
      float closest_saved_candidate_dist = quantile_of(distances_to_candidate, quantile);
      if (closest_saved_candidate_dist >= baseline_distance) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, (int)saved_candidates.size());
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsAryaMountReversed(PriorityQueue& neighbors, int M) const {
    if (neighbors.size() < M) {
      return;
    }

    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first > b.first;  // Descending order
              });

    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;  // Already the distance to the query
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);
      if (closest_saved_candidate_dist >= baseline_distance) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, (int)saved_candidates.size());
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsProbabilisticRank(PriorityQueue& neighbors, int M,
                                        float rank_prune_factor) const {
    // RPF should be set equal to 1.0 or so.
    if (neighbors.size() < M) {
      return;
    }
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (neighbors.size() > 0) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    //Sort by distance to the new node.
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order
              });
    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    std::vector<dist_node_t> saved_candidates;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);
    for (int rank = 0; rank < all_candidates.size(); ++rank) {
      auto [distance, node_id] = all_candidates[rank];
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, node_id);
      if (closest_saved_candidate_dist >= distance) {
        float prune_probability =
            rank_prune_factor *
            (static_cast<float>(rank) / static_cast<float>(all_candidates.size()));
        // Generate a random number between 0 and 1.  If we were doing this a lot, we could
        // make this a static thread_local variable.
        float random_number = distrib(gen);
        if (random_number > prune_probability) {
          saved_candidates.push_back({distance, node_id});
        }
      }
    }

    int loop_limit = std::min(M, static_cast<int>(saved_candidates.size()));
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsNeighborhoodOverlap(PriorityQueue& neighbors, int M,
                                          float overlap_threshold) const {
    if (neighbors.size() < M) {
      return;
    }
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (neighbors.size() > 0) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    //Sort by distance to query.
    std::sort(
        all_candidates.begin(), all_candidates.end(),
        [](const dist_node_t& a, const dist_node_t& b) { return a.first < b.first; });

    std::vector<dist_node_t> saved_candidates;
    std::vector<std::unordered_set<node_id_t>>
        saved_neighbor_sets;  // Store neighbor sets

    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);

      // Get the neighbor set of the current candidate
      std::unordered_set<node_id_t> candidate_neighbor_set;
      node_id_t* candidate_links = _allocator.getNodeLinks(candidate.second);
      for (size_t i = 0; i < M; ++i) {
        if (candidate_links[i] != candidate.second) {  // Avoid self loops
          candidate_neighbor_set.insert(candidate_links[i]);
        }
      }

      float max_overlap = 0.0f;
      // Calculate Jaccard Index
      auto jaccard_index = [](const std::unordered_set<node_id_t>& set1,
                              const std::unordered_set<node_id_t>& set2) {
        if (set1.empty() || set2.empty()) {
          return 0.0f;
        }
        size_t intersection_size = 0;
        for (const auto& element : set1) {
          if (set2.count(element)) {
            intersection_size++;
          }
        }
        size_t union_size = set1.size() + set2.size() - intersection_size;
        return static_cast<float>(intersection_size) / static_cast<float>(union_size);
      };

      for (const auto& saved_set : saved_neighbor_sets) {
        float overlap = jaccard_index(candidate_neighbor_set, saved_set);
        max_overlap = std::max(max_overlap, overlap);
      }

      if (max_overlap < overlap_threshold &&
          closest_saved_candidate_dist >= baseline_distance) {
        saved_candidates.push_back(candidate);
        saved_neighbor_sets.push_back(candidate_neighbor_set);  // Add to saved sets
      }
    }

    int loop_limit = std::min(M, static_cast<int>(saved_candidates.size()));
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsGeometricMean(PriorityQueue& neighbors, int M) const {
    if (neighbors.size() < M) {
      return;
    }
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());
    while (neighbors.size() > 0) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order
              });

    std::vector<dist_node_t> saved_candidates;

    auto geometric_mean_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                               node_id_t node_id) {
      if (saved.empty())
        return std::numeric_limits<float>::max();
      float product = 1.0;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        product *= dist;
      }
      return float(std::pow(product, 1.0 / saved.size()));
    };
    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;
      float closest_saved_candidate_dist =
          geometric_mean_distance_to_node(saved_candidates, candidate.second);
      if (closest_saved_candidate_dist >= baseline_distance) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, static_cast<int>(saved_candidates.size()));
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsAryaMountShuffled(PriorityQueue& neighbors, int M) const {
    if (neighbors.size() < M) {
      return;
    }

    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    // SHUFFLE the candidates instead of sorting
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_candidates.begin(), all_candidates.end(), g);

    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;  // Already the distance to the query
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);
      if (closest_saved_candidate_dist >= baseline_distance) {
        saved_candidates.push_back(candidate);
      }
    }

    int loop_limit = std::min(M, (int)saved_candidates.size());
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsOneSpanner(PriorityQueue& neighbors, int M) const {
    if (neighbors.size() < M) {
      return;
    }

    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order
              });

    std::unordered_set<node_id_t> one_hop_neighborhood;
    for (const auto& candidate : all_candidates) {  // Sorted from close to far.
      bool node_not_reachable =
          (one_hop_neighborhood.find(candidate.second) == one_hop_neighborhood.end());
      // one_hop_neighborhood.contains(candidate.second) // Requires C++20
      if (node_not_reachable) {
        // Node is not present in out-degree neighborhood so add it to the link list.
        saved_candidates.push_back(candidate);
        node_id_t* candidate_links = _allocator.getNodeLinks(candidate.second);
        for (size_t i = 0; i < M; ++i) {
          one_hop_neighborhood.insert(candidate_links[i]);
        }
      }
    }
    int loop_limit = std::min(M, (int)saved_candidates.size());
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }

  void selectNeighborsAryaMountPlusSpanner(PriorityQueue& neighbors, int M) const {
    if (neighbors.size() < M) {
      return;
    }

    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());

    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending order
              });

    auto min_distance_to_node = [&](const std::vector<dist_node_t>& saved,
                                    node_id_t node_id) {
      float min_dist = std::numeric_limits<float>::max();
      if (saved.empty())
        return min_dist;
      for (const auto& saved_node : saved) {
        float dist = _distance.distance(_allocator.getNodeData(saved_node.second),
                                        _allocator.getNodeData(node_id));
        min_dist = std::min(min_dist, dist);
      }
      return min_dist;
    };

    std::unordered_set<node_id_t> one_hop_neighborhood;
    for (const auto& candidate : all_candidates) {
      float baseline_distance = candidate.first;  // Already the distance to the query
      float closest_saved_candidate_dist =
          min_distance_to_node(saved_candidates, candidate.second);
      bool node_not_reachable =
          (one_hop_neighborhood.find(candidate.second) == one_hop_neighborhood.end());
      if ((closest_saved_candidate_dist >= baseline_distance) || node_not_reachable) {
        saved_candidates.push_back(candidate);
        node_id_t* candidate_links = _allocator.getNodeLinks(candidate.second);
        for (size_t i = 0; i < M; ++i) {
          one_hop_neighborhood.insert(candidate_links[i]);
        }
      }
    }

    int loop_limit = std::min(M, (int)saved_candidates.size());
    for (int i = 0; i < loop_limit; i++) {
      neighbors.emplace(saved_candidates[i].first, saved_candidates[i].second);
    }
  }
};

}  // namespace flatnav