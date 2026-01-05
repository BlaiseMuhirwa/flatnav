#pragma once

#include <cstdint>

namespace flatnav {

/**
 * @brief Strategies for two-pass index construction.
 *
 * Two-pass construction builds a cheap initial graph (Pass 1) to collect
 * statistics, then rebuilds with strategy-specific modifications (Pass 2).
 */
enum class TwoPassStrategy {
  /// Standard single-pass construction (no two-pass)
  NONE,

  /// Collect k-occurrence (in-degree) in Pass 1, penalize high-hubness nodes
  /// in Pass 2's neighbor selection to prevent hub concentration
  HUBNESS_SCORING,

  /// Score edges by distance rank and angular diversity in Pass 1,
  /// use scores to guide pruning decisions in Pass 2
  EDGE_QUALITY_SCORING,

  /// Compute node centrality from Pass 1 graph, reinsert nodes in Pass 2
  /// in order of increasing centrality (peripheral nodes first)
  INSERTION_ORDER_OPT
};

/**
 * @brief Candidate search method for Pass 2.
 */
enum class Pass2CandidateMethod {
  /// Use beam search to find candidates (original method, slower but more thorough)
  BEAM_SEARCH,

  /// Use neighbor-of-neighbor expansion (faster at scale, O(M^2) per node)
  NEIGHBOR_EXPANSION
};

/**
 * @brief Configuration for two-pass construction.
 */
struct TwoPassConfig {
  /// Which strategy to use
  TwoPassStrategy strategy = TwoPassStrategy::NONE;

  /// Maximum edges per node in Pass 1 (cheap graph)
  int M_pass1 = 4;

  /// Maximum edges per node in Pass 2 (final graph)
  int M_pass2 = 4;

  /// ef_construction for Pass 1 (low value for speed)
  int ef_construction_pass1 = 10;

  /// ef_construction for Pass 2 (higher value for quality)
  int ef_construction_pass2 = 100;

  /// Number of random entry points for search initialization
  int num_initializations = 100;

  /// Weight for hubness penalty in HUBNESS_SCORING strategy
  /// Higher values penalize high-hubness nodes more aggressively
  float hubness_penalty_weight = 0.1f;

  /// Threshold for edge quality in EDGE_QUALITY_SCORING strategy
  /// Edges with quality below this are more likely to be pruned
  float edge_quality_threshold = 0.5f;

  /// Number of threads for parallel construction
  uint32_t num_threads = 1;

  /// Method for finding Pass 2 candidates
  /// BEAM_SEARCH: Original method, slower but explores more graph
  /// NEIGHBOR_EXPANSION: Faster O(M^2) method using 2-hop neighborhood
  Pass2CandidateMethod pass2_candidate_method = Pass2CandidateMethod::BEAM_SEARCH;

  /// For NEIGHBOR_EXPANSION: number of hops to explore (2 or 3)
  int neighbor_expansion_hops = 2;
};

} // namespace flatnav
