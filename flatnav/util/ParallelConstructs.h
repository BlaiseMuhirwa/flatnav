#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace flatnav {

template <typename Function>
void parallelFor(uint32_t start, uint32_t end, uint32_t num_threads,
                 Function fn) {
  if (num_threads == 0) {
    throw std::invalid_argument("Invalid number of threads");
  }

  // This needs to be an atomic because mutliple threads will be
  // modifying it concurrently.
  std::atomic<uint32_t> current(start);
  std::thread thread_objects[num_threads];

  auto parallel_executor = [&] {
    while (true) {
      uint32_t current_vector_idx = current.fetch_add(1);
      if (current_vector_idx >= end) {
        break;
      }
      fn(current_vector_idx);
    }
  };

  for (uint32_t id = 0; id < num_threads; id++) {
    thread_objects[id] = std::thread(parallel_executor);
  }
  for (uint32_t id = 0; id < num_threads; id++) {
    thread_objects[id].join();
  }
}

} // namespace flatnav