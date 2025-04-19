#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Allocator.h>
#include <flatnav/index/PrimitiveTypes.h>
#include <flatnav/util/Datatype.h>
#include <Eigen/Dense>  // For MatrixXd, VectorXd, determinant
#include <algorithm>
#include <optional>
#include <random>
#include <vector>

namespace flatnav {

using flatnav::distances::DistanceInterface;
using flatnav::distances::InnerProductDistance;
using flatnav::distances::SquaredL2Distance;

/**
 * @brief Class template for selecting pruning heuristics in nearest neighbor search.
 * 
 * This class implements various pruning strategies to select a subset of M neighbors 
 * from a larger candidate set. The pruning heuristics help maintain graph quality
 * while reducing connectivity.
 *
 * @tparam dist_t The distance type used for similarity calculations
 * @tparam label_t The label type associated with graph nodes
 *
 * The class provides different pruning methods including:
 * - Kernel Herding: Uses Random Fourier Features to approximate RBF kernel
 * - DPP (Determinantal Point Process): Selects diverse subsets based on similarity
 *
 * Each pruning method can be configured with different parameters through the select() 
 * method.
 */

template <typename dist_t, typename label_t>
struct PruningHeuristicSelector {
  using dist_label_t = flatnav::index_dist_label_t<label_t>;
  using dist_node_t = flatnav::index_dist_node_t;
  using PriorityQueue = flatnav::index_priority_queue_t;
  using node_id_t = flatnav::index_node_id_t;

  // TODO: Make these references const
  flatnav::FlatMemoryAllocator<int>& _allocator;
  DistanceInterface<dist_t>& _distance;

  InnerProductDistance<flatnav::util::DataType::float32> _ip_distance;

  PruningHeuristicSelector(flatnav::FlatMemoryAllocator<int>& allocator,
                           DistanceInterface<dist_t>& distance)
      : _allocator(allocator), _distance(distance) {}

  /**
   * Parameter now includes a vector of floats. It's size should be at most 3.
   */
  void select(PruningHeuristic heuristic, PriorityQueue& neighbors, int M,
              char* p = nullptr,
              std::optional<std::vector<float>> parameter = std::nullopt) const {
    auto param_size = parameter ? parameter->size() : 0;
    if (param_size > 3) {
      throw std::invalid_argument("Parameter size exceeds maximum allowed size of 3.");
    }
    switch (heuristic) {
      // case PruningHeuristic::DIRECTIONAL_DIVERSITY: {
      //   static_assert(p != nullptr,
      //                 "Composite diversity heuristic requires the p parameter.");
      //   selectNeighborsDiretionalDiversity(neighbors, M, p);
      //   break;
      // }
      // case PruningHeuristic::COMPOSITE_DIVERSITY: {
      //   static_assert(p != nullptr,
      //                 "Composite diversity heuristic requires the p parameter.");
      //   selectNeighborsCompositeDiversity(neighbors, M, p);
      //   break;
      // }
      case PruningHeuristic::KERNEL_HERDING: {
        float gamma = 10.0f;
        int D_rff = 4096;
        selectNeighborsKernelHerding(neighbors, M, p, gamma, D_rff);
        break;
      }
      case PruningHeuristic::DPP: {
        auto beta = parameter.value()[0];
        auto gamma = parameter.value()[1];
        auto mcmc_steps_multiplier = static_cast<int>(parameter.value()[2]);
        selectNeighborsDPP(neighbors, M, p, beta, gamma, mcmc_steps_multiplier);
        break;
      }
      case PruningHeuristic::ARYA_MOUNT: {
        selectNeighborsAryaMount(neighbors, M, p);
        break;
      }
      case PruningHeuristic::ARYA_MOUNT_SANITY_CHECK:
        selectNeighborsAryaMountSanityCheck(neighbors, M, p);
        break;
      case PruningHeuristic::ARYA_MOUNT_SIGMOID_ON_REJECTS: {
        // sigmoid slope = 0.1.
        auto steepness = parameter.has_value() ? parameter.value()[0] : 0.1f;
        selectNeighborsAryaMountSigmoidOnRejects(neighbors, M, p, steepness);
        break;
      }
      case PruningHeuristic::ARYA_MOUNT_RANDOM_ON_REJECTS: {
        auto probability = parameter.has_value() ? parameter.value()[0] : 0.01f;
        selectNeighborsAryaMountRandomOnRejects(neighbors, M, p, probability);
        break;
      }
      case PruningHeuristic::ARYA_MOUNT_REVERSED:
        selectNeighborsAryaMountReversed(neighbors, M, p);
        break;
      case PruningHeuristic::ARYA_MOUNT_SHUFFLED:
        selectNeighborsAryaMountShuffled(neighbors, M, p);
        break;
      case PruningHeuristic::ARYA_MOUNT_PLUS_SPANNER:
        selectNeighborsAryaMountPlusSpanner(neighbors, M, p);
        break;
      case PruningHeuristic::VAMANA: {
        auto alpha = parameter.has_value() ? parameter.value()[0] : 0.8333f;
        selectNeighborsVamana(neighbors, M, p, alpha);
        break;
      }
      case PruningHeuristic::NEAREST_M:
        selectNeighborsPickTopM(neighbors, M, p);
        break;
      case PruningHeuristic::FURTHEST_M:
        selectNeighborsPickTopM(neighbors, M, p, /* nearest = */ false);
        break;
      case PruningHeuristic::CHEAP_OUTDEGREE_CONDITIONAL: {
        int cheap_outgoing_edge_threshold = parameter.has_value()
                                                ? static_cast<int>(parameter.value()[0])
                                                : static_cast<int>(M);
        selectNeighborsCheapOutDegreeConditional(neighbors, M, p,
                                                 cheap_outgoing_edge_threshold);
        break;
      }
      case PruningHeuristic::LARGE_OUTDEGREE_CONDITIONAL: {
        int well_connected_outgoing_edge_threshold =
            parameter.has_value() ? static_cast<int>(parameter.value()[0])
                                  : static_cast<int>(M);
        selectNeighborsLargeOutDegreeConditional(neighbors, M, p,
                                                 well_connected_outgoing_edge_threshold);
        break;
      }
      case PruningHeuristic::SIGMOID_RATIO: {
        auto sigmoid_steepness = parameter.has_value() ? parameter.value()[0]
                                                       : 1.0f;  // Default steepness value
        selectNeighborsSigmoidRatio(neighbors, M, p, sigmoid_steepness);
        break;
      }
      case PruningHeuristic::MEDIAN_ADAPTIVE:
        selectNeighborsMedianAdaptive(neighbors, M, p);
        break;
      case PruningHeuristic::TOP_M_MEDIAN_ADAPTIVE:
        selectNeighborsTopMMeanAdaptive(neighbors, M, p);
        break;
      case PruningHeuristic::MEAN_SORTED_BASELINE:
        selectNeighborsMeanSortedBaseline(neighbors, M, p);
        break;
      case PruningHeuristic::QUANTILE_NOT_MIN: {
        auto quantile = parameter.has_value() ? parameter.value()[0]
                                              : 0.2f;  // Default quantile value
        selectNeighborsQuantileNotMin(neighbors, M, p, quantile);
        break;
      }
      case PruningHeuristic::PROBABILISTIC_RANK: {
        auto rank_prune_factor = parameter.has_value()
                                     ? parameter.value()[0]
                                     : 1.0f;  // Default rank prune factor
        selectNeighborsProbabilisticRank(neighbors, M, p, rank_prune_factor);
        break;
      }
      case PruningHeuristic::NEIGHBORHOOD_OVERLAP: {
        auto overlap_threshold = parameter.has_value()
                                     ? parameter.value()[0]
                                     : 0.8f;  // Default overlap threshold
        selectNeighborsNeighborhoodOverlap(neighbors, M, p, overlap_threshold);
        break;
      }
      case PruningHeuristic::GEOMETRIC_MEAN:
        selectNeighborsGeometricMean(neighbors, M, p);
        break;
      case PruningHeuristic::ONE_SPANNER:
        selectNeighborsOneSpanner(neighbors, M, p);
        break;
      default:
        selectNeighborsAryaMount(neighbors, M, p);
    }
  }

  // void selectNeighborsDiretionalDiversity(PriorityQueue& neighbors, int M,
  //                                         char* p = nullptr, float gamma = 0.9) const {
  //   /**
  //    * Instead of using the Arya & Mount heuristic, which is a magnitude-based diversity criterion,
  //    * we use an angular/directional diversity criterion.
  //    */
  //   if (neighbors.size() <= M)
  //     return;

  //   std::priority_queue<std::pair<float, node_id_t>> candidates;
  //   std::vector<dist_node_t> saved_candidates;
  //   saved_candidates.reserve(M);

  //   while (neighbors.size() > 0) {
  //     auto [distance, id] = neighbors.top();

  //     candidates.emplace(-distance, id);
  //     neighbors.pop();
  //   }

  //   while (candidates.size() > 0) {
  //     if (saved_candidates.size() >= M) {
  //       break;
  //     }
  //     // Extract the closest element from candidates.
  //     auto [distance_to_query, current_node_id] = candidates.top();
  //     distance_to_query = -distance_to_query;
  //     candidates.pop();

  //     // Compute the vector (x - y) for the current candidate and
  //     // for each saved candidate
  //     // and check if the angle between them is less than gamma.
  //     // If so, we keep the current candidate.
  //     // Otherwise, we discard it.
  //     flatnav::util::DataType type = _distance.getDataType();
  //     T* casted_data =
  //         reinterpret_cast<flatnav::util::type_for_data_type(type)::type*>(p);
  //     const void* current_node_data = _allocator.getNodeData(current_node_id);

  //     // Now allocated a T* array to store (casted_data - current_node_data)
  //     T* vec_difference = new T[_distance.dimension()];
  //     for (size_t i = 0; i < _distance.dimension(); ++i) {
  //       vec_difference[i] = casted_data[i] - static_cast<const T*>(current_node_data)[i];
  //     }

  //     bool should_keep_candidate = true;
  //     for (const auto& [_, second_pair_node_id] : saved_candidates) {
  //       // Now compute (x - y) for the saved candidate
  //       const void* saved_candidate_data = _allocator.getNodeData(second_pair_node_id);
  //       T* saved_vec_difference = new T[_distance.dimension()];
  //       for (size_t i = 0; i < _distance.dimension(); ++i) {
  //         saved_vec_difference[i] =
  //             casted_data[i] - static_cast<const T*>(saved_candidate_data)[i];
  //       }

  //       // Compute the angle between the two vectors
  //       float dot_product = 0.0f;
  //       for (size_t i = 0; i < _distance.dimension(); ++i) {
  //         dot_product += vec_difference[i] * saved_vec_difference[i];
  //       }
  //       float norm_current = 0.0f;
  //       float norm_saved = 0.0f;
  //       for (size_t i = 0; i < _distance.dimension(); ++i) {
  //         norm_current += vec_difference[i] * vec_difference[i];
  //         norm_saved += saved_vec_difference[i] * saved_vec_difference[i];
  //       }
  //       norm_current = std::sqrt(norm_current);
  //       norm_saved = std::sqrt(norm_saved);
  //       float angle = std::acos(dot_product / (norm_current * norm_saved));

  //       // Check if the angle is greater than gamma to reject the candidate.
  //       // This means that the candidate is similar to something we already have.
  //       if (angle > gamma) {
  //         should_keep_candidate = false;
  //         // Dealloc first
  //         delete[] saved_vec_difference;
  //         break;  // No need to check other saved candidates
  //       }
  //     }
  //     delete[] vec_difference;

  //     if (should_keep_candidate) {
  //       // We could do neighbors.emplace except we have to iterate
  //       // through saved_candidates, and std::priority_queue doesn't
  //       // support iteration (there is no technical reason why not).
  //       auto current_pair = std::make_pair(-distance_to_query, current_node_id);
  //       saved_candidates.push_back(current_pair);
  //     }
  //   }
  //   // TODO: implement my own priority queue, get rid of vector
  //   // saved_candidates, add directly to neighborqueue earlier.
  //   for (const dist_node_t& current_pair : saved_candidates) {
  //     neighbors.emplace(-current_pair.first, current_pair.second);
  //   }
  // }

  // void selectNeighborsCompositeDiversity(PriorityQueue& neighbors, int M,
  //                                        char* p = nullptr, float gamma = 0.9) const {}

  /**
   * Helper function to retrieve Eigen vector from raw data pointer.
   */
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>> getEigenVector(
      const void* data, size_t dimension) const {
    return Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>>(
        static_cast<const float*>(data), dimension);
  }

  /**
   * @brief Implements the greedy kernel herding algorithm as described in the following papers:
   * 1. Distribution-based sketching of single-cell samples
   *  Ref: https://arxiv.org/pdf/2207.00584
   * 2. Super samples from kernel herding
   *  Ref: https://arxiv.org/pdf/2303.05944
   * 
   * Instead of using the naive implementation using the RBF kernel, we use the 
   * Random Fourier Features (RFF) to approximate the RBF kernel. 
   * This corresponds to algorithm 2 in paper 1.
   * 
   * @param gamma: the bandwidth of the RBF kernel.
   * @param D_rff: the dimension of the RFF. This must be even.
   */
  void selectNeighborsKernelHerding(PriorityQueue& neighbors, int M, char* p = nullptr,
                                    float gamma = 10, int D_rff = 4096) const {
    if (neighbors.size() <= M)
      return;
    if (D_rff <= 0 || D_rff % 2 != 0) {
      throw std::invalid_argument("D_rff must be even and positive.");
    }

    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());
    while (!neighbors.empty()) {
      // Kernel herding doesn't directly use distance to query, but we keep this
      // for reconstructing the final set of neighbors at the end of the greeedy algorithm.
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }
    size_t N = all_candidates.size();
    auto dimension = _distance.dimension();

    // --- Set up the RFF matrix ---
    int D_half = D_rff / 2;

    // 1. Draw random fourier frequencies (matrix W from paper 1)
    // W should be d x (D_rff / 2) according to text, but φ_W(x) uses W^T x.
    // Let's make W size (D_half x original_dim) so W^T x is easy with Eigen.
    // Sample elements W_ij ~ N(0, 1/gamma) according to Algorithm 2 image.
    // Note: Standard RFF for exp(-gamma ||x-y||^2) often uses N(0, 2*gamma).
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> W = Eigen::MatrixXf::Zero(D_half, dimension);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(1.0f / gamma));
    for (int i = 0; i < D_half; ++i) {
      for (int j = 0; j < dimension; ++j) {
        W(i, j) = dist(gen);
      }
    }

    // --- Compute the RFF features for all candidates ---
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> rff_features(N);
    const float scale_factor = std::sqrt(2.0f / D_rff);
    Eigen::Matrix<float, Eigen::Dynamic, 1> projections(D_half);
    Eigen::Matrix<float, Eigen::Dynamic, 1> cos_part(D_half);
    Eigen::Matrix<float, Eigen::Dynamic, 1> sin_part(D_half);

    for (size_t i = 0; i < N; i++) {
      const void* candidate_data = _allocator.getNodeData(all_candidates[i].second);
      auto candidate_eigen = getEigenVector(candidate_data, dimension);

      // Compute the projection: Wx (result is D_half x 1)
      projections = W * candidate_eigen;
      // Compute the cosine and sine parts using Eigen's coefficient-wise operations
      cos_part = projections.array().cos();
      sin_part = projections.array().sin();

      // Concatenate [cos(Wx), sin(Wx)] (result is D_rff x 1)
      rff_features[i].resize(D_rff);
      rff_features[i] << cos_part, sin_part;
      rff_features[i] *= scale_factor;
    }

    // -- Initialize the herding state ---
    // theta_0 = (1/N) * sum_{i=1}^{N} rff_features[i]
    // Using double for summation stability.
    Eigen::Matrix<double, Eigen::Dynamic, 1> theta_0 =
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(D_rff);
    for (size_t i = 0; i < N; i++) {
      theta_0 += rff_features[i].template cast<double>();
    }
    theta_0 /= static_cast<double>(N);

    // current_theta corresponds to theta_{j-1} in Algorithm 2's loop
    Eigen::Matrix<double, Eigen::Dynamic, 1> current_theta = theta_0;

    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);
    std::unordered_set<node_id_t> selected_node_ids;

    // ---iterative greedy selection algorithm---
    for (size_t j = 1; j <= M; j++) {
      double max_objective_value = -std::numeric_limits<double>::infinity();
      int best_candidate_idx = -1;
      // Find argmax theta_{j-1}^T * phi_W(x_i) over x_i not in X_hat
      for (size_t idx = 0; idx < N; idx++) {
        if (selected_node_ids.count(all_candidates[idx].second) > 0) {
          continue;
        }

        // Obejctive: dot product in the feature space (RKHS)
        double objective_value =
            current_theta.dot(rff_features[idx].template cast<double>());
        if (objective_value > max_objective_value) {
          max_objective_value = objective_value;
          best_candidate_idx = idx;
        }
      }

      if (best_candidate_idx == -1) {
        // Add best candidate x_hat to selection
        node_id_t selected_node_id = all_candidates[best_candidate_idx].second;
        saved_candidates.push_back(all_candidates[best_candidate_idx]);
        selected_node_ids.insert(selected_node_id);

        // Update theta for the next iteration: theta_j = theta_{j-1} + theta_0 - phi_W(x_hat)
        current_theta +=
            theta_0 - rff_features[best_candidate_idx].template cast<double>();
      } else {
        // No valid candidate found, break
        break;
      }
    }

    // --- Reconstruct the final set of neighbors ---
    for (const auto& selected_node : saved_candidates) {
      neighbors.emplace(selected_node.first, selected_node.second);
    }
  }

  void selectNeighborsDPP(PriorityQueue& neighbors, int M, char* p = nullptr,
                          float beta = 1.0f, float gamma = 1.0f,
                          int mcmc_steps_multiplier = 10) const {
    if (neighbors.size() <= M) {
      return;
    }

    std::vector<dist_node_t> all_candidates;
    all_candidates.reserve(neighbors.size());
    while (!neighbors.empty()) {
      all_candidates.push_back(neighbors.top());
      neighbors.pop();
    }

    // DPP works best when candidates are somewhat ordered, although not strictly required.
    // Sorting by distance to query is a reasonable first step.
    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const dist_node_t& a, const dist_node_t& b) {
                return a.first < b.first;  // Ascending distance to query
              });

    size_t N = all_candidates.size();

    // --- 1. Construct the DPP Kernel L ---
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(N, N);
    std::vector<double> quality(N);  // Use double for stability

    // Precompute quality and node data pointers
    std::vector<const void*> candidate_data(N);
    for (size_t i = 0; i < N; ++i) {
      // Use double precision for potentially small exponential values
      quality[i] = static_cast<double>(
          exp(-beta * all_candidates[i].first * all_candidates[i].first));
      candidate_data[i] = _allocator.getNodeData(all_candidates[i].second);
      L(i, i) = quality[i];  // Set diagonal
    }

    // Compute off-diagonal elements based on pairwise distances
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        // TODO: Consider caching distance computations if they become a bottleneck
        float dist_ij = _distance.distance(candidate_data[i], candidate_data[j]);
        double similarity_ij = static_cast<double>(exp(-gamma * dist_ij * dist_ij));
        double val = std::sqrt(quality[i] * quality[j]) * similarity_ij;
        L(i, j) = val;
        L(j, i) = val;  // Kernel is symmetric
      }
    }

    // --- 2. Approximate k-DPP Sampling (MCMC) ---
    std::vector<size_t> current_selection_indices(N);
    std::iota(current_selection_indices.begin(), current_selection_indices.end(),
              0);  // 0, 1, ..., N-1

    // Initial state: Select the first M indices (closest M points)
    std::vector<size_t> current_S_indices(current_selection_indices.begin(),
                                          current_selection_indices.begin() + M);
    std::vector<size_t> current_complement_indices(current_selection_indices.begin() + M,
                                                   current_selection_indices.end());

    std::random_device rd;
    std::mt19937 gen(rd());

    // Calculate initial determinant efficiently if possible, otherwise compute directly
    Eigen::MatrixXd L_S = L(current_S_indices, current_S_indices);
    double det_L_S = L_S.determinant();

    // Add small epsilon to determinants to avoid division by zero or issues with det=0
    const double det_epsilon = 1e-30;

    int total_mcmc_steps = mcmc_steps_multiplier * N;  // Heuristic number of steps

    for (int step = 0; step < total_mcmc_steps; ++step) {
      if (current_S_indices.empty() || current_complement_indices.empty()) {
        break;  // Cannot swap
      }

      // Choose random element i from S and j from complement(S)
      std::uniform_int_distribution<> dist_S(0, current_S_indices.size() - 1);
      std::uniform_int_distribution<> dist_comp(0, current_complement_indices.size() - 1);
      size_t idx_in_S = dist_S(gen);
      size_t idx_in_comp = dist_comp(gen);

      size_t i = current_S_indices[idx_in_S];              // Actual index in L
      size_t j = current_complement_indices[idx_in_comp];  // Actual index in L

      // Propose new set S' = S - {i} + {j}
      std::vector<size_t> proposed_S_indices = current_S_indices;
      proposed_S_indices[idx_in_S] = j;  // Replace i with j

      // Calculate determinant of L_S'
      // WARNING: Naive determinant calculation is O(M^3).
      // Optimization: Use determinant update formulas (matrix determinant lemma)
      // for O(M^2) or specialized DPP libraries.
      // For simplicity here, we recalculate.
      Eigen::MatrixXd L_S_prime = L(proposed_S_indices, proposed_S_indices);
      double det_L_S_prime = L_S_prime.determinant();

      // Acceptance probability
      double acceptance_prob = 0.0;
      if (std::abs(det_L_S) > det_epsilon) {  // Avoid division by zero/small numbers
        acceptance_prob = std::min(1.0, std::abs(det_L_S_prime / det_L_S));
      } else if (std::abs(det_L_S_prime) > det_epsilon) {
        acceptance_prob = 1.0;  // Accept if new determinant is non-zero and old was zero
      }
      // If both are near zero, probability is effectively 0 or 1 depending on exact zeros, treat as 0 here.

      std::uniform_real_distribution<> accept_dist(0.0, 1.0);
      if (accept_dist(gen) < acceptance_prob) {
        // Accept the swap
        current_S_indices =
            proposed_S_indices;  // This is O(M), could optimize swap if needed
        std::swap(current_complement_indices[idx_in_comp],
                  current_complement_indices.back());
        current_complement_indices.pop_back();
        current_complement_indices.push_back(i);

        det_L_S = det_L_S_prime;  // Update determinant
      }
    }

    // --- 3. Populate neighbors queue with the final selected set ---
    for (size_t selected_idx : current_S_indices) {
      neighbors.emplace(all_candidates[selected_idx].first,
                        all_candidates[selected_idx].second);
    }

    // Ensure exactly M neighbors if MCMC somehow resulted in fewer (shouldn't happen with this swap logic)
    // Or trim if somehow more (also shouldn't happen)
    // This part is likely unnecessary with the fixed-size swap MCMC but acts as a safeguard.
    while (neighbors.size() > M) {
      neighbors
          .pop();  // Assumes largest distances are popped first (correct for std::priority_queue max-heap)
    }
    // If somehow neighbors.size() < M, this implementation doesn't add more back.
    // This indicates an issue in the MCMC or initial conditions.
  }

  /**
   * @brief Selects neighbors from the PriorityQueue, according to the original
   * HNSW heuristic from Arya&Mount. The neighbors priority queue contains
   * elements sorted by distance where the top element is the furthest neighbor
   * from the query.
   */
  void selectNeighborsAryaMount(PriorityQueue& neighbors, int M,
                                char* p = nullptr) const {
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

  void selectNeighborsAryaMountSanityCheck(PriorityQueue& neighbors, int M,
                                           char* p = nullptr) const {
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

  void selectNeighborsVamana(PriorityQueue& neighbors, int M, char* p = nullptr,
                             float alpha = 1.0f) const {
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

  void selectNeighborsPickTopM(PriorityQueue& neighbors, int M, char* p = nullptr,
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
                                                char* p = nullptr,
                                                int cheap_outgoing_edge_threshold = 10) const {
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
      PriorityQueue& neighbors, int M, char* p = nullptr,
      int well_connected_outgoing_edge_threshold = 10) const {
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

  void selectNeighborsSigmoidRatio(PriorityQueue& neighbors, int M, char* p = nullptr,
                                   float steepness = 1.0f) const {
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
                                                char* p = nullptr,
                                                float steepness = 1.0f) const {
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
                                               char* p = nullptr,
                                               float accept_anyway_prob = 0.5f) const {
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

  void selectNeighborsMedianAdaptive(PriorityQueue& neighbors, int M,
                                    char* p = nullptr) const {
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

  void selectNeighborsTopMMeanAdaptive(PriorityQueue& neighbors, int M,
                                       char* p = nullptr) const {
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

  void selectNeighborsMeanSortedBaseline(PriorityQueue& neighbors, int M,
                                         char* p = nullptr) const {
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

  void selectNeighborsQuantileNotMin(PriorityQueue& neighbors, int M, char* p = nullptr,
                                     double quantile = 0.5) const {
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

  void selectNeighborsAryaMountReversed(PriorityQueue& neighbors, int M,
                                        char* p = nullptr) const {
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
                                        char* p = nullptr,
                                        float rank_prune_factor = 1.0f) const {
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
                                          char* p = nullptr,
                                          float overlap_threshold = 0.5f) const {
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

  void selectNeighborsGeometricMean(PriorityQueue& neighbors, int M,
                                    char* p = nullptr) const {
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

  void selectNeighborsAryaMountShuffled(PriorityQueue& neighbors, int M,
                                        char* p = nullptr) const {
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

  void selectNeighborsOneSpanner(PriorityQueue& neighbors, int M,
                                 char* p = nullptr) const {
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

  void selectNeighborsAryaMountPlusSpanner(PriorityQueue& neighbors, int M,
                                           char* p = nullptr) const {
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