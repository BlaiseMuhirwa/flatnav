#pragma once

namespace flatnav {

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
}