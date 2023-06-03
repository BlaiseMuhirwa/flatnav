#pragma once
#include "../DistanceInterface.h"
#include <cstddef> // for size_t

// This is the base distance function implementation for the L2 distance on
// floating-point inputs. We provide specializations that use SIMD when
// supported by the compiler and compatible with the input _dimension. These
// specializations inherit from SquaredL2Distance and overload only the
// distance_impl method.

namespace flatnav {

class SquaredL2Distance : public DistanceInterface<SquaredL2Distance> {
  friend class DistanceInterface<SquaredL2Distance>;
  static const int DISTANCE_ID = 0;

public:
  SquaredL2Distance(size_t dim) {
    _dimension = dim;
    _data_size_bytes = dim * sizeof(float);
  }

private:
  size_t _dimension;
  size_t _data_size_bytes;

  float distance_impl(const void *x, const void *y) {
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

  size_t data_size_impl() { return _data_size_bytes; }

  void transform_data_impl(void *dst, const void *src) {
    std::memcpy(dst, src, _data_size_bytes);
  }

  void serialize_impl(std::ofstream &out) {
    // TODO: Make this safe across machines and compilers.
    out.write(reinterpret_cast<const char *>(&DISTANCE_ID), sizeof(int));
    out.write(reinterpret_cast<char *>(&_dimension), sizeof(size_t));
  }

  void deserialize(std::ifstream &in) {
    // TODO: Make this safe across machines and compilers.
    int distance_id_check;
    in.read(reinterpret_cast<char *>(&distance_id_check), sizeof(int));
    if (distance_id_check != DISTANCE_ID) {
      throw std::invalid_argument(
          "Error reading distance metric: Distance ID does not match "
          "the ID of the deserialized distance instance.");
    }
    in.read(reinterpret_cast<char *>(&_dimension), sizeof(size_t));
    _data_size_bytes = _dimension * sizeof(float);
  }
};

} // namespace flatnav