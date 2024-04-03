#pragma once

#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/distances/SquaredL2Extensions.h>

namespace flatnav {

struct SquaredL2DistanceFactory {

  static std::shared_ptr<SquaredL2Distance> create(size_t dim) {
    // Check if AVX512 is supported
    if (dim % 16 == 0) {
      return std::make_shared<SquaredL2Avx512>(dim);
    } else {
      return std::make_shared<SquaredL2Distance>(dim);
    }
  }
};

} // namespace flatnav