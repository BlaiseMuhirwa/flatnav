

#pragma once

#include <cstdint>
#include <vector>

namespace flatnav::quantization {

class CentroidsGenerator {
public:
  CentroidsGenerator(uint64_t dim, uint64_t num_centroids,
                     uint32_t num_iterations = 25,
                     uint32_t max_points_per_centroid = 256,
                     bool normalized = true, bool verbose = false)
      : _dim(dim), _num_centroids(num_centroids),
        _clustering_iterations(num_iterations),
        _max_points_per_centroid(max_points_per_centroid),
        _normalized(normalized), _verbose(verbose), _seed(3333) {}

  /**
   * @brief Run k-means clustering algorithm to compute D-dimensional
   * centroids given n D-dimensional vectors.

   * The algorithm proceeds as follows:
   * - Select k datapoints as the initial centroids (we use a random
   initialization for now).
   *   We can also use another strategy such as kmeans++ initialization due to
   Arthur and
   *   Vassilvitskii.
   https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
   *
   * - Assign each data point to its nearest centroid based on l2 distances.
   * - Update each centroid to be the mean of its assigned datapoints
   * - Repeat steps 2 & 3 until we reach _num_iterations
   *
   * @param vectors       the input datapoints
   * @param vec_weights   weight associated with each datapoint: NULL or size n
   * @param n             The number of datapoints
   */
  void generateCentroids(const float *vectors, const float *vec_weights,
                         uint64_t n);

  inline std::vector<float> &centroids() { return _centroids; }

private:
  uint64_t _dim;
  uint64_t _num_centroids;

  std::vector<float> _centroids;

  // Number of clustering iterations
  uint32_t _clustering_iterations;
  // normalize centroids if set to true

  // limit the dataset size. If the number of datapoints
  // exceeds k * _max_points_per_centroid, we use subsampling
  uint32_t _max_points_per_centroid;
  bool _normalized;
  bool _verbose;

  // seed for random number generator;
  int _seed;
};

} // namespace flatnav::quantization