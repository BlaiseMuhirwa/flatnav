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
  if (num_threads <= 0) {
    throw std::invalid_argument("Invalid number of threads");
  }

  if (num_threads == 1) {
    for (uint32_t i = start; i < end; i++) {
      fn(i, 0);
    }
    return;
  }
  std::vector<std::thread> threads;
  std::atomic<uint32_t> current(start);

  std::exception_ptr last_exception = nullptr;
  std::mutex last_exception_mutex;

  for (uint32_t thread_id = 0; thread_id < num_threads; thread_id++) {
    threads.push_back(std::thread([&, thread_id] {
      while (true) {
        uint32_t current_value = current.fetch_add(1);
        if (current_value >= end) {
          break;
        }

        try {
          fn(current_value, thread_id);
        } catch (...) {
          std::unique_lock<std::mutex> lock(last_exception_mutex);
          last_exception = std::current_exception();

          current = end;
          break;
        }
      }
    }));
  }

  for (auto &thread : threads) {
    thread.join();
  }

  if (last_exception) {
    std::rethrow_exception(last_exception);
  }
}

} // namespace flatnav