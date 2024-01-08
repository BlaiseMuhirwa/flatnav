#pragma once

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cstddef> // for size_t
#include <cstring> // for memcpy
#include <flatnav/DistanceInterface.h>
#include <flatnav/util/SIMDDistanceSpecializations.h>
#include <functional>
#include <iostream>
#include <limits>

namespace flatnav {

// This is the base distance function implementation for inner product distances
// on floating-point inputs.

class InnerProductDistance : public DistanceInterface<InnerProductDistance> {

  friend class DistanceInterface<InnerProductDistance>;
  // Enum for compile-time constant
  enum { DISTANCE_ID = 1 };

public:
  InnerProductDistance() = default;

  explicit InnerProductDistance(size_t dim)
      : _dimension(dim), _data_size_bytes(dim * sizeof(float)),
        _distance_computer(
            [this](const void *x, const void *y, const size_t &dimension) {
              return this->defaultDistanceImpl(x, y, dimension);
            }) {
    setDistanceFunction();
  }

  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const {
    (void)asymmetric;
    return _distance_computer(x, y, _dimension);
  }

private:
  size_t _dimension;
  size_t _data_size_bytes;
  std::function<float(const void *, const void *, const size_t &)>
      _distance_computer;

  friend class cereal::access;

  template <typename Archive> void serialize(Archive &ar) {
    ar(_dimension);

    // If loading, we need to set the data size bytes
    if (Archive::is_loading::value) {
      _data_size_bytes = _dimension * sizeof(float);
      _distance_computer = [this](const void *x, const void *y,
                                  const size_t &dimension) {
        return this->defaultDistanceImpl(x, y, dimension);
      };

      setDistanceFunction();
    }
  }

  inline size_t getDimension() const { return _dimension; }

  size_t dataSizeImpl() { return _data_size_bytes; }

  void transformDataImpl(void *dst, const void *src) {
    std::memcpy(dst, src, _data_size_bytes);
  }

  void getSummaryImpl() {
    std::cout << "\nInnerProductDistance Parameters" << std::flush;
    std::cout << "\n-----------------------------"
              << "\n"
              << std::flush;
    std::cout << "Dimension: " << _dimension << "\n" << std::flush;
  }

  void setDistanceFunction() {
#ifndef NO_MANUAL_VECTORIZATION
#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
    _distance_computer = distanceImplInnerProductSIMD16ExtSSE;
#if defined(USE_AVX512)
    if (platform_supports_avx512()) {
      _distance_computer = distanceImplInnerProductSIMD16ExtAVX512;
    } else if (platform_supports_avx()) {
      _distance_computer = distanceImplInnerProductSIMD16ExtAVX;
    }
#elif defined(USE_AVX)
    if (platform_supports_avx()) {
      _distance_computer = distanceImplInnerProductSIMD16ExtAVX;
    }
#endif
    if (!(_dimension % 16 == 0)) {
      if (_dimension % 4 == 0) {
#if defined(USE_AVX)
        _distance_computer = distanceImplInnerProductSIMD4ExtAVX;
#else
        // TODO: This conditional branch is untested.
        _distance_computer = distanceImplInnerProductSIMD4ExtSSE;
#endif
      } else if (_dimension > 16) {
        _distance_computer = distanceImplInnerProductSIMD16ExtResiduals;
      } else if (_dimension > 4) {
        _distance_computer = distanceImplInnerProductSIMD4ExtResiduals;
      }
    }

#endif // USE_AVX512 || USE_AVX || USE_SSE
#endif // NO_MANUAL_VECTORIZATION
  }

  float defaultDistanceImpl(const void *x, const void *y,
                            const size_t &dimension) const {
    // Default implementation of inner product distance, in case we cannot
    // support the SIMD specializations for special input _dimension sizes.
    float *p_x = (float *)x;
    float *p_y = (float *)y;
    float result = 0;
    for (size_t i = 0; i < dimension; i++) {
      result += p_x[i] * p_y[i];
    }
    return 1.0 - result;
  }
};

} // namespace flatnav