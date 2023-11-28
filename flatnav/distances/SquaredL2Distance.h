#pragma once
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cstddef> // for size_t
#include <cstring> // for memcpy
#include <flatnav/DistanceInterface.h>
#include <flatnav/util/SIMDIntrinsics.h>
#include <iostream>

// This is the base distance function implementation for the L2 distance on
// floating-point inputs. We provide specializations that use SIMD when
// supported by the compiler and compatible with the input _dimension.

namespace flatnav {

class SquaredL2Distance : public DistanceInterface<SquaredL2Distance> {

  friend class DistanceInterface<SquaredL2Distance>;
  enum { DISTANCE_ID = 0 };

public:
  SquaredL2Distance() = default;
  explicit SquaredL2Distance(size_t dim)
      : _dimension(dim), _data_size_bytes(dim * sizeof(float)),
        _distance_computer(&SquaredL2Distance::defaultDistanceImpl) {

#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
    _distance_computer = &flatnav::util::distanceImplSquaredL2SIMD16ExtSSE;
#if defined(USE_AVX512)
    if (flatnav::util::platform_supports_avx512()) {
      _distance_computer = &flatnav::util::distanceImplSquaredL2SIMD16ExtAVX512;
    } else if (platform_supports_avx()) {
      _distance_computer = &flatnav::util::distanceImplSquaredL2SIMD16ExtAVX;
    }
#elif defined(USE_AVX)
    if (flatnav::util::platform_supports_avx()) {
      _distance_computer = &flatnav::util::distanceImplSquaredL2SIMD16ExtAVX;
    }
#endif

    if (!(_dimension % 16 == 0) && _dimension % 4 == 0) {
      _distance_computer = &flatnav::util::distanceImplSquaredL2SIMD4Ext;
    } else if (_dimension > 16) {
      _distance_computer =
          &flatnav::util::distanceImplSquaredL2SIMD16ExtResiduals;
    } else if (_dimension > 4) {
      _distance_computer =
          &flatnav::util::distanceImplSquaredL2SIMD4ExtResiduals;
    }

#endif
  }

  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const {
    (void)asymmetric;
    return _distance_computer(x, y, _dimension);
  }

private:
  size_t _dimension;
  size_t _data_size_bytes;
  float (*_distance_computer)(const void *, const void *, size_t &) const;

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

  float defaultDistanceImpl(const void *x, const void *y,
                            size_t &dimension) const {
    // Default implementation of squared-L2 distance, in case we cannot
    // support the SIMD specializations for special input _dimension sizes.
    float *p_x = (float *)x;
    float *p_y = (float *)y;
    float squared_distance = 0;

    for (size_t i = 0; i < dimension; i++) {
      float difference = *p_x - *p_y;
      p_x++;
      p_y++;
      squared_distance += difference * difference;
    }
    return squared_distance;
  }
};

} // namespace flatnav
