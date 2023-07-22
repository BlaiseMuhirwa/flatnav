#pragma once

#include "verifysimd.h"

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cstring>
#include <iostream>
#include <stdint.h>

namespace flatnav {

class ExplicitSet {
private:
  uint32_t _mark;
  uint32_t *_table;
  uint32_t _tableSize;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(_mark, _tableSize);

    if (Archive::is_loading::value) {
      // If we are loading, allocate memory for the table and delete
      // previously allocated memory if any.
      delete[] _table;
      _table = new uint32_t[_tableSize];
    }

    archive(cereal::binary_data(_table, _tableSize * sizeof(uint32_t)));
  }

public:
  ExplicitSet() = default;

  ExplicitSet(const uint32_t size) : _mark(0), _table(NULL), _tableSize(size) {
    // initialize values to 0
    _table = new uint32_t[_tableSize]();
  }

  inline void prefetch(const uint32_t num) const {
#ifdef USE_SSE
    _mm_prefetch((char *)_table[num], _MM_HINT_T0);
#endif
  }

  inline void insert(const uint32_t num) { set(num); }

  inline void set(const uint32_t num) { _table[num] = _mark; }

  inline void reset(const uint32_t num) { _table[num] = _mark + 1; }

  inline void clear() { _mark++; }

  inline bool operator[](const uint32_t num) { return (_table[num] == _mark); }

  ~ExplicitSet() { delete[] _table; }

  // copy constructor
  ExplicitSet(const ExplicitSet &other) {
    _tableSize = other._tableSize;
    _mark = other._mark;
    delete[] _table;
    _table = new uint32_t[_tableSize];
    std::memcpy(other._table, _table, _tableSize * sizeof(uint32_t));
  }

  // move constructor
  ExplicitSet(ExplicitSet &&other) noexcept {
    _tableSize = other._tableSize;
    _mark = other._mark;
    _table = other._table;
    other._table = NULL;
    other._tableSize = 0;
    other._mark = 0;
  }

  // copy assignment
  ExplicitSet &operator=(const ExplicitSet &other) {
    return *this = ExplicitSet(other);
  }

  // move assignment
  ExplicitSet &operator=(ExplicitSet &&other) noexcept {
    _tableSize = other._tableSize;
    _mark = other._mark;
    _table = other._table;
    other._table = NULL;
    other._tableSize = 0;
    other._mark = 0;
    return *this;
  }
};

} // namespace flatnav