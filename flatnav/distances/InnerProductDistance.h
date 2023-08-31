#pragma once

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cstddef> // for size_t
#include <flatnav/DistanceInterface.h>
#include <limits>

// This is the base distance function implementation for inner product distances
// on floating-point inputs.

namespace flatnav {

class InnerProductDistance : public DistanceInterface<InnerProductDistance> {
  friend class DistanceInterface<InnerProductDistance>;
  // Enum for compile-time constant
  enum { DISTANCE_ID = 1 };

public:
  explicit InnerProductDistance(size_t dim) {
    _dimension = dim;
    _data_size_bytes = dim * sizeof(float);
  }

private:
  size_t _dimension;
  size_t _data_size_bytes;

  friend class cereal::access;

  InnerProductDistance() = default;

  template <typename Archive> void serialize(Archive &archive) {
    archive(_dimension, _data_size_bytes);
  }

  inline size_t getDimension() const { return _dimension; }

  float distanceImpl(const void *x, const void *y) {
    // Default implementation of inner product distance, in case we cannot
    // support the SIMD specializations for special input _dimension sizes.
    float *p_x = (float *)x;
    float *p_y = (float *)y;
    float result = 0;
    for (size_t i = 0; i < _dimension; i++) {
      result += p_x[i] * p_y[i];
    }
    return 1.0 - result;
  }

  size_t dataSizeImpl() { return _data_size_bytes; }

  void transformDataImpl(void *dst, const void *src) {
    std::memcpy(dst, src, _data_size_bytes);
  }
};

} // namespace flatnav