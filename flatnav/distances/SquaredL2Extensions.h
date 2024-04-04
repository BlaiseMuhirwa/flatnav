#pragma once

#include <flatnav/DistanceInterface.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/util/Macros.h>
#include <flatnav/util/SimdAvx.h>

using flatnav::util::simd16float32;
using flatnav::util::simd8float32;

namespace flatnav {

class SquaredL2Avx2 : public SquaredL2Distance {
public:
  explicit SquaredL2Avx2(size_t dim) : SquaredL2Distance(dim) {}


  float distanceImpl(const void*x, const void *y, bool asymmetric = false) const override {
    (void) asymmetric;

    float *pointer_x = static_cast<float *>(const_cast<void *>(x));
    float *pointer_y = static_cast<float *>(const_cast<void *>(y));

    const float* end_x = pointer_x + (_dimension >> 4 << 4);
    simd8float32 difference, v1, v2;
    simd8float32 sum(0.0f);

    while (pointer_x != end_x) {
      v1.loadu(pointer_x);
      v2.loadu(pointer_y);
      difference = v1 - v2;
      sum += difference * difference;
      pointer_x += 8;
      pointer_y += 8;
    }

    return sum.reduce_add();
  }

};

class SquaredL2Avx512 : public SquaredL2Distance {

public:
  explicit SquaredL2Avx512(size_t dim) : SquaredL2Distance(dim) {}

  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const override {
    (void)asymmetric;

    float *pointer_x = static_cast<float *>(const_cast<void *>(x));
    float *pointer_y = static_cast<float *>(const_cast<void *>(y));

    // Align to 16-floats boundary
    const float *end_x = pointer_x + (_dimension >> 4 << 4);
    simd16float32 difference, v1, v2;

    simd16float32 sum(0.0f);

    while (pointer_x != end_x) {
      v1.loadu(pointer_x);
      v2.loadu(pointer_y);
      difference = v1 - v2;
      sum += difference * difference;
      pointer_x += 16;
      pointer_y += 16;
    }
    return sum.reduce_add();
  }
};

} // namespace flatnav