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
 * NOTE: It is the responsibility of the caller to validate the arguments.
 *     For instance, before invoking this function, the caller should ensure
 *     that the number of threads is not 0 and if checkpointing is enabled,
 *     the checkpoint interval and checkpoint function are valid.
 */
template <typename Function, typename... Args>
void executeInParallel(
    uint32_t start_index, uint32_t end_index, uint32_t num_threads,
    Function function, bool enable_checkpointing = false,
    float checkpoint_every = 0.1, // Every 10% of the steps
    std::function<void(const std::string &)> checkpoint_fn = nullptr,
    const std::string &checkpoint_filepath = "", Args... additional_args) {
  if (num_threads == 0) {
    throw std::invalid_argument("Invalid number of threads");
  }

  // This needs to be an atomic because mutliple threads will be
  // modifying it concurrently.
  std::atomic<uint32_t> current(start_index);
  std::thread thread_objects[num_threads];
  std::mutex checkpt_mutex;

  auto parallel_executor = [&] {
    while (true) {
      uint32_t current_vector_idx = current.fetch_add(1);
      if (current_vector_idx >= end_index) {
        break;
      }
      // Use std::apply to pass arguments to the function
      std::apply(function, std::tuple_cat(std::make_tuple(current_vector_idx),
                                          std::make_tuple(additional_args...)));

      if (enable_checkpointing) {
        if (current_vector_idx %
                static_cast<uint32_t>(checkpoint_every * end_index) ==
            0) {
          std::lock_guard<std::mutex> lock(checkpt_mutex);
          checkpoint_fn(checkpoint_filepath);

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

} // namespace flatnav