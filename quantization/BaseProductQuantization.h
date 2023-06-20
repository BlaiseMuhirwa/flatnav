#pragma once

#include "../flatnav/DistanceInterface.h"
#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace flatnav {

/** This is the main class definition for a regular Product Quantizer
 * The implementation is simultaneously inspired by FAISS(https://faiss.ai)
 * and the paper Cache Locality is not Enough: High-Performance Nearest
 * Neighbor Search with Product Quantization Fast Scan
 * http://www.vldb.org/pvldb/vol9/p288-andre.pdf
 *
 */

template <typename dist_t> class ProductQuantizer {

public:
  ProductQuantizer() = default;

  /** PQ Constructor.
   *
   * @param dim      dimensionality of the input vectors
   * @param M        number of subquantizers
   * @param nbits    number of bit per subvector index
   *
   * TODO: Only pass the distance interface to the underlying index.
   * This will be possible once the PQ integration with the flatnav
   * index is complete.
   */
  ProductQuantizer(std::unique_ptr<DistanceInterface<dist_t>> dist,
                   uint32_t dim, uint32_t M, uint32_t nbits)
      : _num_subquantizers(M), _num_bits(nbits), _distance(std::move(dist)) {}

  // Return the centroids associated with subvector m
  const float *getCentroids(uint32_t m, uint32_t i) const {
    auto index = m * _centroids_count + i * _subvector_dim;
    return &_centroids[index];
  }

  // Quantize a single vector with PQ
  void computePQCode(const float *vector, uint8_t *code) const;

  // Quantize multiple vectors with PQ
  void computePQCodes(const float *vectors, uint8_t *codes, uint32_t n);

  // Trains product quantization on a set of vectors
  void train(size_t n, const float *vectors);

  // Decode a single vector from a give code
  void decode(const uint8_t *code, float *vector) const;

  // Decode multiple vectors given their respective codes.
  void decode(const uint8_t *code, float *vectors, size_t n) const;

  /** Compute distance table for a single vector.
   * For more information on distance tables, see
   * http://www.vldb.org/pvldb/vol9/p288-andre.pdf
   *
   * @param x         input vector size d
   * @param dis_table output table, size (_num_subquantizers x _centroids_count)
   */
  void computeDistanceTable(const float *vector, float *dist_table) const;

  void computeDistanceTables(const float *vectors, float *dist_tables,
                             size_t n) const;

  /** perform a search
   * @param query_vectors        query vectors, size num_queries * d
   * @param num_queries          number of queries
   * @param codes                PQ codes, size ncodes * code_size
   * @param ncodes               nb of vectors
   */
  std::vector<float> search(const float *query_vectors, size_t num_queries,
                            const uint8_t *codes, const size_t ncodes) const;

  inline uint32_t getNumSubquantizers() const { return _num_subquantizers; }

  inline uint32_t getNumBitsPerIndex() const { return _num_bits; }

  inline uint32_t getCentroidsCount() const { return _centroids_count; }

private:
  // This is typically called M in the literature, but we will
  // name it differently to differentiate M from HNSW (max edges per node)
  // and M in product quantization (number of subquantizers)
  uint32_t _num_subquantizers;

  // Number of bits per quantization index.
  uint32_t _num_bits;

  // The dimension of each subvector
  uint32_t _subvector_dim;

  // The number of centroids per subquantizer.
  // NOTE: If `_num_bits` = k, we can have maximum of 2^k
  // `_centroids_count`. This simply follows from the fact that each
  // computed index can only index in [0, 2^k - 1] range.
  uint32_t _centroids_count;

  // Centroid table. It will have size
  // (_num_subquanitizers x _centroids_count x _subvector_dim)
  std::vector<float> _centroids;

  // Represents centroids in a transposed form. This is useful while performing
  // the Asymmetric Distance Computation (ADC) where we are able to use the
  // transposed centroids to leverage SIMD instructions.
  // Layout: (_subvector_dim x _num_subquantizers x _centroids_count)
  std::vector<float> _transposed_centroids;

  // Initialization
  enum TrainType {
    DEFAULT,
    HOT_START,     // The centroids are already initialized
    SHARED,        // Share dictionary accross PQ segments
    HYPERCUBE,     // Initialize centroids with nbits-D hypercube
    HYPERCUBE_PCA, // Initialize centroids with nbits-D hypercube
  };

  TrainType _train_type;
  std::unique_ptr<DistanceInterface<dist_t>> _distance;
};

} // namespace flatnav