#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>

namespace flatnav {

/**
 * @brief Variadic template for executing a function in parallel using STL's
 * threading library. This is preferred in lieu of OpenMP only because it will
 * not require having logic for installing OpenMP on the host system while
 * installing the Python library.
 */
template <typename Function, typename... Args>
void executeInParallel(uint32_t start_index, uint32_t end_index, uint32_t num_threads, Function function,
                       Args... additional_args) {
  if (num_threads == 0) {
    throw std::invalid_argument("Invalid number of threads");
  }

  std::atomic<uint32_t> current(start_index);
  std::thread thread_objects[num_threads];

  // // Calculate total range and initialize progress tracking
  // const uint32_t total_range = end_index - start_index;
  // std::atomic<uint32_t> next_progress_milestone(10);

  auto parallel_executor = [&] {
    while (true) {
      uint32_t current_vector_idx = current.fetch_add(1);
      if (current_vector_idx >= end_index) {
        break;
      }
      // Invoke the function with arguments
      std::apply(function,
                 std::tuple_cat(std::make_tuple(current_vector_idx), std::make_tuple(additional_args...)));

      // // Check progress milestone
      // uint32_t progress = ((current_vector_idx - start_index + 1) * 100) / total_range;
      // uint32_t milestone = next_progress_milestone.load();
      // if (progress >= milestone) {
      //   if (next_progress_milestone.compare_exchange_strong(milestone, milestone + 10)) {
      //     std::cout << "Progress: " << milestone << "% completed\n";
      //   }
      // }
    }
  };

  for (uint32_t id = 0; id < num_threads; id++) {
    thread_objects[id] = std::thread(parallel_executor);
  }
  for (uint32_t id = 0; id < num_threads; id++) {
    thread_objects[id].join();
  }
}

}  // namespace flatnav


