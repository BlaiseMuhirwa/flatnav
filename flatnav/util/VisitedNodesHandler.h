#pragma once

#include <flatnav/util/SIMDDistanceSpecializations.h>

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <vector>

namespace flatnav {

class VisitedNodesHandler {
private:
  uint32_t _mark;
  uint32_t *_table;
  uint32_t _table_size;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(_mark, _table_size);

    if (Archive::is_loading::value) {
      // If we are loading, allocate memory for the table and delete
      // previously allocated memory if any.
      delete[] _table;
      _table = new uint32_t[_table_size];
    }

    archive(cereal::binary_data(_table, _table_size * sizeof(uint32_t)));
  }

public:
  VisitedNodesHandler() = default;

  VisitedNodesHandler(const uint32_t size) : _mark(0), _table_size(size) {
    // initialize values to 0
    _table = new uint32_t[_table_size]();
  }

  inline void prefetch(const uint32_t num) const {
#ifdef USE_SSE
    _mm_prefetch((char *)_table[num], _MM_HINT_T0);
#endif
  }

  inline uint32_t getMark() const { return _mark; }

  inline void insert(const uint32_t num) { set(num); }

  inline void set(const uint32_t num) { _table[num] = _mark; }

  inline uint32_t size() const { return _table_size; }

  inline void reset(const uint32_t num) { _table[num] = _mark + 1; }

  inline void clear() { _mark++; }

  inline bool operator[](const uint32_t num) { return (_table[num] == _mark); }

  ~VisitedNodesHandler() { delete[] _table; }

  // copy constructor
  VisitedNodesHandler(const VisitedNodesHandler &other) {
    _table_size = other._table_size;
    _mark = other._mark;
    _table = new uint32_t[_table_size];
    std::memcpy(_table, other._table, _table_size * sizeof(uint32_t));
  }

  // move constructor
  VisitedNodesHandler(VisitedNodesHandler &&other) noexcept {
    _table_size = other._table_size;
    _mark = other._mark;
    _table = other._table;
    other._table = NULL;
    other._table_size = 0;
    other._mark = 0;
  }

  // copy assignment
  VisitedNodesHandler &operator=(const VisitedNodesHandler &other) {
    if (this != &other) {
      delete[] _table;
      _table_size = other._table_size;
      _mark = other._mark;
      _table = new uint32_t[_table_size];
      std::memcpy(_table, other._table, _table_size * sizeof(uint32_t));
    }
    return *this;
  }

  // move assignment
  VisitedNodesHandler &operator=(VisitedNodesHandler &&other) noexcept {
    _table_size = other._table_size;
    _mark = other._mark;
    _table = other._table;
    other._table = NULL;
    other._table_size = 0;
    other._mark = 0;
    return *this;
  }
};

class ThreadSafeVisitedNodesHandler {
  std::vector<std::unique_ptr<VisitedNodesHandler>> _handler_pool;
  std::mutex _pool_guard;
  uint32_t _num_elements;
  uint32_t _total_handlers_in_use;

  friend class cereal::access;

  template <typename Archive> void serialize(Archive &archive) {
    archive(_handler_pool, _num_elements, _total_handlers_in_use);
  }

public:
  ThreadSafeVisitedNodesHandler() = default;
  ThreadSafeVisitedNodesHandler(uint32_t initial_pool_size,
                                uint32_t num_elements)
      : _handler_pool(initial_pool_size), _num_elements(num_elements),
        _total_handlers_in_use(1) {
    for (uint32_t handler_id = 0; handler_id < _handler_pool.size();
         handler_id++) {
      _handler_pool[handler_id] =
          std::make_unique<VisitedNodesHandler>(/* size = */ _num_elements);
    }
  }

  VisitedNodesHandler *pollAvailableHandler() {
    std::unique_lock<std::mutex> lock(_pool_guard);

    if (!_handler_pool.empty()) {
      // NOTE: release() call is required here to ensure that we don't free
      // the handler's memory before using it since it's under a unique pointer.
      auto *handler = _handler_pool.back().release();
      _handler_pool.pop_back();
      return handler;
    } else {
      // TODO: This is not great because it assumes the caller is responsible
      // enough to return this handler to the pool. If the caller doesn't return
      // the handler to the pool, we will have a memory leak. This can be
      // resolved by std::unique_ptr but I prefer to use a raw pointer here.
      auto *handler = new VisitedNodesHandler(/* size = */ _num_elements);
      _total_handlers_in_use++;
      return handler;
    }
  }

  void pushHandler(VisitedNodesHandler *handler) {
    std::unique_lock<std::mutex> lock(_pool_guard);

    _handler_pool.push_back(std::make_unique<VisitedNodesHandler>(*handler));
    _handler_pool.shrink_to_fit();
  }

  inline uint32_t getPoolSize() { return _handler_pool.size(); }

  ~ThreadSafeVisitedNodesHandler() = default;
};

} // namespace flatnav