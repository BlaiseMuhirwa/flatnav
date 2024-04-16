#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <tuple>
#include <utility>

namespace flatnav::util {

/**
 * @brief Variadic template for executing a function in parallel using STL's
 * threading library. This is preferred in lieu of OpenMP only because it will
 * not require having logic for installing OpenMP on the host system while
 * installing the Python library.
 */
template <typename Function, typename... Args>
void executeInParallel(uint32_t start_index, uint32_t end_index,
                       uint32_t num_threads, Function function,
                       std::function<void(double)> progress_callback = nullptr,
                       Args... additional_args) {
  if (num_threads == 0) {
    throw std::invalid_argument("Invalid number of threads");
  }

  // This needs to be an atomic because mutliple threads will be
  // modifying it concurrently.
  std::atomic<uint32_t> current(start_index);
  std::atomic<uint32_t> progress_counter(start_index);
  uint32_t total_items = end_index - start_index;
  std::thread thread_objects[num_threads];

  auto parallel_executor = [&] {
    while (true) {
      uint32_t current_vector_idx = current.fetch_add(1);
      if (current_vector_idx >= end_index) {
        break;
      }
      // Use std::apply to pass arguments to the function
      std::apply(function, std::tuple_cat(std::make_tuple(current_vector_idx),
                                          std::make_tuple(additional_args...)));

      // Update the progress counter
      if (progress_callback) {
        uint32_t current_progress = progress_counter.fetch_add(1) + 1;
        // Update every 10% of the progress
        if (current_progress % (total_items / 10) == 0) {
          progress_callback(static_cast<double>(current_progress) /
                            total_items * 100.0);
        }
      }
    }
  };

  for (uint32_t id = 0; id < num_threads; id++) {
    thread_objects[id] = std::thread(parallel_executor);
  }
  for (uint32_t id = 0; id < num_threads; id++) {
    thread_objects[id].join();
  }
}

} // namespace flatnav::util