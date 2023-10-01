#pragma once
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cstddef> // for size_t
#include <cstring> // for memcpy
#include <flatnav/DistanceInterface.h>
#include <iostream>

// This is the base distance function implementation for the L2 distance on
// floating-point inputs. We provide specializations that use SIMD when
// supported by the compiler and compatible with the input _dimension. These
// specializations inherit from SquaredL2Distance and overload only the
// distance_impl method.

namespace flatnav {

class SquaredL2Distance : public DistanceInterface<SquaredL2Distance> {

  friend class DistanceInterface<SquaredL2Distance>;
  enum { DISTANCE_ID = 0 };

public:
  SquaredL2Distance() = default;
  explicit SquaredL2Distance(size_t dim) {
    _dimension = dim;
    _data_size_bytes = dim * sizeof(float);
  }

  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const {
    (void)asymmetric;
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

private:
  size_t _dimension;
  size_t _data_size_bytes;

  friend class ::cereal::access;

  template <typename Archive> void serialize(Archive &ar) {
    ar(_dimension);

    // If loading, we need to set the data size bytes
    if (Archive::is_loading::value) {
      _data_size_bytes = _dimension * sizeof(float);
    }
  }

  inline size_t getDimension() const { return _dimension; }

  size_t dataSizeImpl() { return _data_size_bytes; }

  void transformDataImpl(void *destination, const void *src) {
    std::memcpy(destination, src, _data_size_bytes);
  }

  void printParamsImpl() {
    std::cout << "\nSquaredL2Distance Parameters" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    std::cout << "Dimension: " << _dimension << std::endl;
  }
};

} // namespace flatnav
