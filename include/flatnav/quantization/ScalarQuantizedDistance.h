#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/IPDistanceDispatcher.h>
#include <flatnav/distances/L2DistanceDispatcher.h>
#include <flatnav/util/Datatype.h>
#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_set>
#include <stdexcept>
#include <vector>

namespace flatnav::quantization {

using flatnav::distances::DistanceInterface;
using flatnav::distances::IPDistanceDispatcher;
using flatnav::distances::L2DistanceDispatcher;
using flatnav::distances::MetricType;
using flatnav::util::DataType;

/// Scalar quantizer that maps fp32 vectors to int8 using per-dimension
/// min/max uniform quantization (8-bit). 
template <MetricType metric>
class ScalarQuantizedDistance
    : public DistanceInterface<ScalarQuantizedDistance<metric>> {

  friend class DistanceInterface<ScalarQuantizedDistance<metric>>;

 public:
  ScalarQuantizedDistance() = default;

  explicit ScalarQuantizedDistance(size_t dim)
      : _dimension(dim),
        _min_vals(dim, std::numeric_limits<float>::max()),
        _max_vals(dim, std::numeric_limits<float>::lowest()),
        _scales(dim, 0.0f),
        _is_trained(false) {}

  static std::unique_ptr<ScalarQuantizedDistance<metric>> create(size_t dim) {
    return std::make_unique<ScalarQuantizedDistance<metric>>(dim);
  }

  /// Learn per-dimension min/max from training data and derive scales.
  /// @param data               Pointer to row-major fp32 training vectors.
  /// @param n                  Number of training vectors.
  /// @param max_train_samples  If > 0 and < n, sample this many rows for
  ///                           min/max computation instead of scanning all n.
  void train(const float* data, size_t n, size_t max_train_samples = 0) {
    if (n == 0) {
      throw std::invalid_argument(
          "ScalarQuantizedDistance::train: need at least one training vector.");
    }

    std::fill(_min_vals.begin(), _min_vals.end(),
              std::numeric_limits<float>::max());
    std::fill(_max_vals.begin(), _max_vals.end(),
              std::numeric_limits<float>::lowest());

    if (max_train_samples > 0 && max_train_samples < n) {
      // Generate max_train_samples unique random indices (O(max_train_samples) memory)
      std::unordered_set<size_t> idx_set;
      idx_set.reserve(max_train_samples);
      std::mt19937 rng(42);
      std::uniform_int_distribution<size_t> dist(0, n - 1);
      while (idx_set.size() < max_train_samples) {
        idx_set.insert(dist(rng));
      }
      // Compute per-dimension min and max over sampled rows
      for (size_t idx : idx_set) {
        const float* vec = data + idx * _dimension;
        for (size_t d = 0; d < _dimension; d++) {
          if (vec[d] < _min_vals[d]) _min_vals[d] = vec[d];
          if (vec[d] > _max_vals[d]) _max_vals[d] = vec[d];
        }
      }
    } else {
      // Compute per-dimension min and max over all rows
      for (size_t i = 0; i < n; i++) {
        const float* vec = data + i * _dimension;
        for (size_t d = 0; d < _dimension; d++) {
          if (vec[d] < _min_vals[d]) _min_vals[d] = vec[d];
          if (vec[d] > _max_vals[d]) _max_vals[d] = vec[d];
        }
      }
    }

    // Derive scales: scale[d] = 255.0 / (max[d] - min[d])
    for (size_t d = 0; d < _dimension; d++) {
      float range = _max_vals[d] - _min_vals[d];
      if (range < 1e-10f) {
        // avoid division by zero
        _scales[d] = 0.0f;
      } else {
        _scales[d] = 255.0f / range;
      }
    }

    _is_trained = true;
  }

  bool isTrained() const { return _is_trained; }

  /// Quantize a single fp32 vector to int8.
  /// @param src   Pointer to fp32 input vector of length _dimension.
  /// @param dst   Pointer to int8 output buffer of length _dimension.
  void quantize(const float* src, int8_t* dst) const {
    for (size_t d = 0; d < _dimension; d++) {
      float val = (src[d] - _min_vals[d]) * _scales[d];
      // Clamp to [0, 255], then shift to signed int8 range [-128, 127]
      val = std::max(0.0f, std::min(255.0f, std::round(val)));
      dst[d] = static_cast<int8_t>(static_cast<int>(val) - 128);
    }
  }

  /// Dequantize a single int8 vector back to fp32.
  /// @param src   Pointer to int8 input vector of length _dimension.
  /// @param dst   Pointer to fp32 output buffer of length _dimension.
  void dequantize(const int8_t* src, float* dst) const {
    for (size_t d = 0; d < _dimension; d++) {
      if (_scales[d] == 0.0f) {
        dst[d] = _min_vals[d];
      } else {
        dst[d] =
            (static_cast<float>(src[d]) + 128.0f) / _scales[d] + _min_vals[d];
      }
    }
  }

  size_t dimension() const { return _dimension; }

  // Accessors for testing
  const std::vector<float>& minVals() const { return _min_vals; }
  const std::vector<float>& maxVals() const { return _max_vals; }
  const std::vector<float>& scales() const { return _scales; }

 private:
  size_t _dimension = 0;
  std::vector<float> _min_vals;
  std::vector<float> _max_vals;
  std::vector<float> _scales;
  bool _is_trained = false;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(_dimension, _min_vals, _max_vals, _scales, _is_trained);
  }

  inline size_t getDimension() const { return _dimension; }

  /// Returns the stored data size per vector: dim * sizeof(int8_t).
  inline size_t dataSizeImpl() { return _dimension * sizeof(int8_t); }

  /// Input data type is float32 -- this controls how addBatch casts
  /// user data pointers. If we were to use DataType::int8, we would be 
  /// misleading the Index method that adds nodes.
  inline DataType getDataTypeImpl() const { return DataType::float32; }

  /// Quantize fp32 input to int8 for storage.
  inline void transformDataImpl(void* destination, const void* src) {
    /// NOTE: There is no need to do std::memcpy since we are quantizing and writing directly 
    /// to the memory block that was preallocated during index construction. 
    quantize(static_cast<const float*>(src),
             static_cast<int8_t*>(destination));
  }

  /// Compute distance between x and y.
  /// When asymmetric=true: x is a raw fp32 query, y is stored int8.
  ///   -> quantize x on-the-fly, then compute int8-vs-int8.
  /// When asymmetric=false: both x and y are stored int8.
  ///   -> compute int8-vs-int8 directly.
  float distanceImpl(const void* x, const void* y,
                     bool asymmetric = false) const {
    if (asymmetric) {
      // x is float query, y is stored int8.
      // Cache the quantized query by pointer: during beam search the same
      // float query is compared against thousands of stored int8 vectors,
      // so we only need to quantize once per unique query pointer.
      thread_local std::vector<int8_t> query_buf;
      thread_local const float* cached_ptr = nullptr;

      const float* query_ptr = static_cast<const float*>(x);
      if (query_ptr != cached_ptr) {
        if (query_buf.size() != _dimension) {
          query_buf.resize(_dimension);
        }
        quantize(query_ptr, query_buf.data());
        cached_ptr = query_ptr;
      }
      return computeInt8Distance(query_buf.data(),
                                 static_cast<const int8_t*>(y));
    }
    // Both are stored int8
    return computeInt8Distance(static_cast<const int8_t*>(x),
                               static_cast<const int8_t*>(y));
  }

  float computeInt8Distance(const int8_t* a, const int8_t* b) const {
    if constexpr (metric == MetricType::L2) {
      return L2DistanceDispatcher::dispatch<int8_t>(a, b, _dimension);
    } else {
      return IPDistanceDispatcher::dispatch<int8_t>(a, b, _dimension);
    }
  }

  void getSummaryImpl() {
    std::cout << "\nScalarQuantizedDistance Parameters" << std::flush;
    std::cout << "\n-----------------------------\n" << std::flush;
    std::cout << "Metric: "
              << (metric == MetricType::L2 ? "L2" : "IP") << "\n"
              << std::flush;
    std::cout << "Dimension: " << _dimension << "\n" << std::flush;
    std::cout << "Trained: " << (_is_trained ? "yes" : "no") << "\n"
              << std::flush;
  }
};

}  // namespace flatnav::quantization
