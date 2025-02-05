#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/IPDistanceDispatcher.h>
#include <flatnav/util/Datatype.h>
#include <flatnav/util/InnerProductSimdExtensions.h>
#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cstddef>  // for size_t
#include <cstring>  // for memcpy
#include <functional>
#include <iostream>
#include <limits>

namespace flatnav::distances {

// This is the base distance function implementation for inner product distances
// on floating-point inputs.

using util::DataType;
using util::type_for_data_type;

template <DataType data_type = DataType::float32>
class InnerProductDistance : public DistanceInterface<InnerProductDistance<data_type>> {

  friend class DistanceInterface<InnerProductDistance>;
  // Enum for compile-time constant
  enum { DISTANCE_ID = 1 };

 public:
  InnerProductDistance() = default;
  InnerProductDistance(size_t dim)
      : _dimension(dim), _data_size_bytes(dim * flatnav::util::size(data_type)) {}

  static std::unique_ptr<InnerProductDistance<data_type>> create(size_t dim) {
    return std::make_unique<InnerProductDistance<data_type>>(dim);
  }

  constexpr float distanceImpl(const void* x, const void* y, [[maybe_unused]] bool asymmetric = false) const {
    return IPDistanceDispatcher::dispatch(static_cast<const typename type_for_data_type<data_type>::type*>(x),
                                          static_cast<const typename type_for_data_type<data_type>::type*>(y),
                                          _dimension);
  }

  DataType getDataTypeImpl() const { return data_type; }

 private:
  size_t _dimension;
  size_t _data_size_bytes;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(_dimension, _data_size_bytes);
  }

  inline size_t getDimension() const { return _dimension; }

  size_t dataSizeImpl() { return _data_size_bytes; }

  void transformDataImpl(void* dst, const void* src) { std::memcpy(dst, src, _data_size_bytes); }

  void getSummaryImpl() {
    std::cout << "\nInnerProductDistance Parameters" << std::flush;
    std::cout << "\n-----------------------------"
              << "\n"
              << std::flush;
    std::cout << "Dimension: " << _dimension << "\n" << std::flush;
  }
};

}  // namespace flatnav::distances