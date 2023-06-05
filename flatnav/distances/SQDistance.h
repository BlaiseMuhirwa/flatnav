#pragma once
#include "../DistanceInterface.h"
#include <cstddef> // for size_t

// This implements the quantized distance functions from:
// "Low-Precision Quantization for Efficient Nearest Neighbor Search" by
// Ko, Lakshman, Keivanloo and Schkufza (https://arxiv.org/abs/2110.08919).
namespace flatnav {

class SquaredL2Distance : public DistanceInterface<SquaredL2Distance> {
  friend class DistanceInterface<SquaredL2Distance>;
  static const int distance_id = 0;

public:
  SquaredL2Distance(size_t dim) {
    _dimension = dim;
    _data_size_bytes = dim * sizeof(float);
  }

private:
  size_t _dimension;
  size_t _data_size_bytes;

  float distanceImpl(const void *x, const void *y) {
    // Default implementation of squared-L2 distance, in case we cannot
    // support the SIMD specializations for special input _dimension sizes.
    float *p_x = (float *)x;
    float *p_y = (float *)y;
    float squared_distance = 0;

    for (size_t i = 0; i < _dimension; i++) {
      float difference = *p_x - *p_y;
      p_x++;
      p_y++;
      squared_distance += difference * difference;
    }
    return squared_distance;
  }

  size_t dataSizeImpl() { return _data_size_bytes; }

  void transformDataImpl(void *destination, const void *src) {
    std::memcpy(destination, src, _data_size_bytes);
  }

  void serializeImpl(std::ofstream &out) {
    // TODO: Make this safe across machines and compilers.
    out.write(reinterpret_cast<const char *>(&distance_id), sizeof(int));
    out.write(reinterpret_cast<char *>(&_dimension), sizeof(size_t));
  }

  void deserializeImpl(std::ifstream &in) {
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