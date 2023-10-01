#pragma once

#include <flatnav/DistanceInterface.h>
#include <cereal/access.hpp>

namespace flatnav::quantization {

class LowPrecisionQuantization
    : public DistanceInterface<LowPrecisionQuantization> {
  friend class DistanceInterface<LowPrecisionQuantization>;

public:
  LowPrecisionQuantization() = default;

private:

    friend class cereal::access;

    template<typename Archive>
    void serialize(Archive& ar) {
        (void) ar;
    }



}

} // namespace flatnav::quantization
