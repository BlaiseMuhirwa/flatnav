#pragma once
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cstddef> // for size_t
#include <cstring> // for memcpy
#include <flatnav/DistanceInterface.h>
#include <flatnav/util/SIMDDistanceSpecializations.h>
#include <functional>
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
        _distance_computer(std::bind(&SquaredL2Distance::defaultDistanceImpl,
                                     this, std::placeholders::_1,
                                     std::placeholders::_2,
                                     std::placeholders::_3)) {
    setDistanceFunction();
  }

  static std::shared_ptr<SquaredL2Distance> create(size_t dim) {
    // Optimizations will assume that dim > 16. If not, we will use the naive
    // distance implementation
    if (dim < 16) {
      return std::make_shared<SquaredL2Distance>(dim);
    }

#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
#if defined(USE_AVX512)
    if (dim % 16 == 0) {
      return std::make_shared<SquaredL2SIMDAVX512>(dim);
    } else if (dim % 4 == 0) {
      return std::make_shared<SquaredL2SIMDAVX>(dim);
    }
#elif defined(USE_AVX)
    return std::make_shared<SquaredL2SIMDAVX>(dim);
#elif defined(USE_SSE)
    return std::make_shared<SquaredL2SIMDSSE>(dim);
#endif // USE_AVX512
#endif // USE_AVX512 || USE_AVX || USE_SSE
    return std::make_shared<SquaredL2Distance>(dim);
  }

  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const {
    (void)asymmetric;
    return _distance_computer(x, y, _dimension);
  }

protected:
  size_t _dimension;
  size_t _data_size_bytes;
  std::function<float(const void *, const void *, const size_t &)>
      _distance_computer;

  friend class ::cereal::access;

  template <typename Archive> void serialize(Archive &ar) {
    ar(_dimension);

    // If loading, we need to set the data size bytes
    if (Archive::is_loading::value) {
      _data_size_bytes = _dimension * sizeof(float);
      _distance_computer = std::bind(
          &SquaredL2Distance::defaultDistanceImpl, this, std::placeholders::_1,
          std::placeholders::_2, std::placeholders::_3);

      setDistanceFunction();
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

  void setDistanceFunction() {
#ifndef NO_MANUAL_VECTORIZATION
#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
    _distance_computer = distanceImplSquaredL2SIMD16ExtSSE;
#if defined(USE_AVX512)
    if (platform_supports_avx512()) {
      _distance_computer = distanceImplSquaredL2SIMD16ExtAVX512;
    } else if (platform_supports_avx()) {
      _distance_computer = distanceImplSquaredL2SIMD16ExtAVX;
    }
#elif defined(USE_AVX)
    if (platform_supports_avx()) {
      _distance_computer = distanceImplSquaredL2SIMD16ExtAVX;
    }
#endif
    if (!(_dimension % 16 == 0)) {
      if (_dimension % 4 == 0) {
        _distance_computer = distanceImplSquaredL2SIMD4Ext;
      } else if (_dimension > 16) {
        _distance_computer = distanceImplSquaredL2SIMD16ExtResiduals;
      } else if (_dimension > 4) {
        _distance_computer = distanceImplSquaredL2SIMD4ExtResiduals;
      }
    }

#endif
#endif // NO_MANUAL_VECTORIZATION
  }

  float defaultDistanceImpl(const void *x, const void *y,
                            const size_t &dimension) const {
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

#if defined(USE_AVX512)
class SquaredL2SIMDAVX512 : public SquaredL2Distance {
public:
  SquaredL2SIMDAVX512() = default;
  explicit SquaredL2SIMDAVX512(size_t dim) : SquaredL2Distance(dim) {}
  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const {
    (void)asymmetric;
    float *p_x = (float *)(x);
    float *p_y = (float *)(y);

    float PORTABLE_ALIGN64 temp_res[16];
    size_t dimension_1_16 = _dimension >> 4;
    const float *p_end_x = p_x + (dimension_1_16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0.0f);

    while (p_x != p_end_x) {
      v1 = _mm512_loadu_ps(p_x);
      v2 = _mm512_loadu_ps(p_y);
      diff = _mm512_sub_ps(v1, v2);
      sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
      p_x += 16;
      p_y += 16;
    }

    _mm512_store_ps(temp_res, sum);
    return temp_res[0] + temp_res[1] + temp_res[2] + temp_res[3] + temp_res[4] +
           temp_res[5] + temp_res[6] + temp_res[7] + temp_res[8] + temp_res[9] +
           temp_res[10] + temp_res[11] + temp_res[12] + temp_res[13] +
           temp_res[14] + temp_res[15];
  }
};
#endif // USE_AVX512

#if defined(USE_AVX)
class SquaredL2SIMDAVX : public SquaredL2Distance {
public:
  SquaredL2SIMDAVX() = default;
  explicit SquaredL2SIMDAVX(size_t dim) : SquaredL2Distance(dim) {}
  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const {
    (void)asymmetric;
    float *p_x = (float *)(x);
    float *p_y = (float *)(y);

    float PORTABLE_ALIGN32 temp_res[8];
    size_t dimension_1_16 = _dimension >> 4;
    const float *p_end_x = p_x + (dimension_1_16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0.0f);

    while (p_x != p_end_x) {
      v1 = _mm256_loadu_ps(p_x);
      v2 = _mm256_loadu_ps(p_y);
      diff = _mm256_sub_ps(v1, v2);
      sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
      p_x += 8;
      p_y += 8;

      v1 = _mm256_loadu_ps(p_x);
      v2 = _mm256_loadu_ps(p_y);
      diff = _mm256_sub_ps(v1, v2);
      sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
      p_x += 8;
      p_y += 8;
    }

    _mm256_store_ps(temp_res, sum);

    return temp_res[0] + temp_res[1] + temp_res[2] + temp_res[3] + temp_res[4] +
           temp_res[5] + temp_res[6] + temp_res[7];
  }
};
#endif // USE_AVX

} // namespace flatnav
