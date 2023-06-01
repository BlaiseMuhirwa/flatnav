#pragma once

#include <cstddef>  // for size_t
#include <fstream>  // for ifstream, ofstream

namespace flatnav {

// We use the CRTP to implement static polymorphism on the distance. This is
// done to allow for metrics and distance functions that support arbitrary 
// pre-processing (such as quantization etc) without having to call the
// distance function through a pointer or virtual function call.
// CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

template<typename T>
class DistanceInterface {
    public:
    float distance(const void* x, const void* y) {
        // This computes the distance for inputs x and y. If the distance
        // requires a pre-processing transformation (e.g. quantization),
        // then the inputs to distance(x, y) should be pre-transformed.
        return static_cast<T*>(this)->distance_impl(x, y);
    }

    size_t data_size(){
        // Returns the size, in bytes, of the transformed data representation.
        return static_cast<T*>(this)->data_size_impl();
    }

    void transform_data(void* dst, const void* src){
        // This transforms the data located at src into a form that is writeable
        // to disk / storable in RAM. For distance functions that don't
        // compress the input, this just passses through a copy from src to dst.
        // However, there are functions (e.g. with quantization) where the
        // in-memory representation is not the same as the raw input.
        static_cast<T*>(this)->transform_data_impl(dst, src);
    }

    void serialize(std::ofstream& out){
        static_cast<T*>(this)->serialize_impl(out);
    }

    void deserialize(std::ifstream& in){
        static_cast<T*>(this)->deserialize_impl(in);
    }
};

}