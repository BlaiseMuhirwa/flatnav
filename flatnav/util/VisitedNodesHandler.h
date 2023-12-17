#pragma once

#include <flatnav/util/SIMDDistanceSpecializations.h>

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

public:
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
  VisitedNodesHandler(const VisitedNodesHandler &other)
      : _table_size(other._table_size), _mark(other._mark) {

    _table = new uint32_t[_table_size];
    std::memcpy(_table, other._table, _table_size * sizeof(uint32_t));
  }

  // move constructor
  VisitedNodesHandler(VisitedNodesHandler &&other) noexcept
      : _table_size(other._table_size), _mark(other._mark),
        _table(other._table) {
    other._table = nullptr;
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

/**
 *
 * @brief Manages a pool of VisitedNodesHandler objects in a thread-safe manner.
 *
 * This class is designed to efficiently provide and manage a pool of
 * VisitedNodesHandler instances for concurrent use in multi-threaded
 * environments. It ensures that each handler can be used by only one thread at
 * a time without the risk of concurrent access and modification.
 *
 * The class preallocates a specified number of VisitedNodesHandler objects to
 * eliminate the overhead of dynamic allocation during runtime. It uses a mutex
 * to synchronize access to the handler pool, ensuring that only one thread can
 * modify the pool at any given time. This mechanism provides both thread safety
 * and improved performance by reusing handler objects instead of continuously
 * creating and destroying them.
 *
 * When a thread requires a VisitedNodesHandler, it can call
 * `pollAvailableHandler()` to retrieve an available handler from the pool. If
 * the pool is empty, the function will dynamically allocate a new handler to
 * ensure that the requesting thread can proceed with its task. Once the thread
 * has finished using the handler, it should return it to the pool by calling
 * `pushHandler()`.
 *
 * @note The class assumes that all threads will properly return the handlers to
 * the pool after use. Failing to return a handler will deplete the pool and
 * lead to dynamic allocation, negating the performance benefits.
 *
 * Usage example:
 * @code
 * ThreadSafeVisitedNodesHandler handler_pool(10, 1000);
 * VisitedNodesHandler* handler = handler_pool.pollAvailableHandler();
 * // Use the handler in a thread...
 * handler_pool.pushHandler(handler);
 * @endcode
 *
 * @param initial_pool_size The number of handler objects to initially create
 * and store in the pool.
 * @param num_elements The size of each VisitedNodesHandler, which typically
 * corresponds to the number of nodes or elements that each handler is expected
 * to manage.
 */
class ThreadSafeVisitedNodesHandler {
  std::vector<VisitedNodesHandler *> _handler_pool;
  std::mutex _pool_guard;
  uint32_t _num_elements;

public:
  ThreadSafeVisitedNodesHandler(uint32_t initial_pool_size,
                                uint32_t num_elements)
      : _handler_pool(initial_pool_size), _num_elements(num_elements) {
    for (uint32_t handler_id = 0; handler_id < _handler_pool.size();
         handler_id++) {
      _handler_pool[handler_id] =
          new VisitedNodesHandler(/* size = */ _num_elements);
    }
  }

  VisitedNodesHandler *pollAvailableHandler() {
    std::unique_lock<std::mutex> lock(_pool_guard);

    if (!_handler_pool.empty()) {
      auto *handler = _handler_pool.back();
      _handler_pool.pop_back();
      return handler;
    } else {
      return new VisitedNodesHandler(/* size = */ _num_elements);
    }
  }

  void pushHandler(VisitedNodesHandler *handler) {
    std::unique_lock<std::mutex> lock(_pool_guard);

    _handler_pool.push_back(handler);
  }

  inline uint32_t getPoolSize() { return _handler_pool.size(); }

  ~ThreadSafeVisitedNodesHandler() {
    while (!_handler_pool.empty()) {
      auto *handler = _handler_pool.back();
      _handler_pool.pop_back();
      delete handler;
    }
  }
};

} // namespace flatnav