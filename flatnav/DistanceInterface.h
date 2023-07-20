#pragma once

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cstddef> // for size_t
#include <fstream> // for ifstream, ofstream
#include <iostream>

namespace flatnav {

// We use the CRTP to implement static polymorphism on the distance. This is
// done to allow for metrics and distance functions that support arbitrary
// pre-processing (such as quantization etc) without having to call the
// distance function through a pointer or virtual function call.
// CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

template <typename T> class DistanceInterface {
public:
  float distance(const void *x, const void *y) {
    // This computes the distance for inputs x and y. If the distance
    // requires a pre-processing transformation (e.g. quantization),
    // then the inputs to distance(x, y) should be pre-transformed.
    return static_cast<T *>(this)->distanceImpl(x, y);
  }

  size_t dataSize() {
    // Returns the size, in bytes, of the transformed data representation.
    return static_cast<T *>(this)->dataSizeImpl();
  }

  // This transforms the data located at src into a form that is writeable
  // to disk / storable in RAM. For distance functions that don't
  // compress the input, this just passses through a copy from src to
  // destination. However, there are functions (e.g. with quantization) where
  // the in-memory representation is not the same as the raw input.
  void transformData(void *destination, const void *src) {
    static_cast<T *>(this)->transformDataImpl(destination, src);
  }

  void serialize(std::ofstream &out) {
    static_cast<T *>(this)->serializeImpl(out);
  }

  void deserialize(std::ifstream &in) {
    static_cast<T *>(this)->deserializeImpl(in);
  }

private:
  // Private constructor for cereal
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {}
};

} // namespace flatnav