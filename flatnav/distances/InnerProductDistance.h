#pragma once
#include "../DistanceInterface.h"
#include <cstddef> // for size_t

// This is the base distance function implementation for inner product distances
// on floating-point inputs.

namespace flatnav {

class InnerProductDistance : public DistanceInterface<InnerProductDistance> {
  friend class DistanceInterface<InnerProductDistance>;
  static const int distance_id = 1;

public:
  InnerProductDistance(size_t dim) {
    _dimension = dim;
    _data_size_bytes = dim * sizeof(float);
  }

private:
  size_t _dimension;
  size_t _data_size_bytes;

  float distance_impl(const void *x, const void *y) {
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

  size_t data_size_impl() { return _data_size_bytes; }

  void transform_data_impl(void *dst, const void *src) {
    std::memcpy(dst, src, _data_size_bytes);
  }

  void serialize_impl(std::ofstream &out) {
    // TODO: Make this safe across machines and compilers.
    out.write(reinterpret_cast<const char *>(&distance_id), sizeof(int));
    out.write(reinterpret_cast<char *>(&_dimension), sizeof(size_t));
  }

  void deserialize(std::ifstream &in) {
    // TODO: Make this safe across machines and compilers.
    int distance_id_check;
    in.read(reinterpret_cast<char *>(&distance_id_check), sizeof(int));
    if (distance_id_check != distance_id) {
      throw std::invalid_argument(
          "Error reading distance metric: Distance ID does not match "
          "the ID of the deserialized distance instance.");
    }
    in.read(reinterpret_cast<char *>(&_dimension), sizeof(size_t));
    _data_size_bytes = _dimension * sizeof(float);
  }
};

} // namespace flatnav