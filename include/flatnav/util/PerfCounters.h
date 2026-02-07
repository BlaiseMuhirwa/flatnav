#pragma once

/**
 * @file PerfCounters.h
 * @brief Lightweight hardware performance counter instrumentation for Linux.
 *
 * This provides inline measurement of cache misses, TLB misses, and other
 * hardware events for specific code sections. Used to validate cache locality
 * hypotheses in entry point selection (anchor vs random initialization).
 *
 * Usage:
 *   // At program start (once per thread that will use counters)
 *   flatnav::perf::CounterGroup counters;
 *   counters.start();
 *
 *   // Around code of interest
 *   auto snapshot = counters.read();
 *   // ... code to measure ...
 *   auto delta = counters.readDelta(snapshot);
 *
 *   // Accumulate into named stats
 *   stats.accumulate("entry_point_selection", delta);
 *
 * Compile with -DFLATNAV_PERF_COUNTERS to enable. Without this flag,
 * all operations become no-ops with zero overhead.
 */

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef FLATNAV_PERF_COUNTERS
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace flatnav {
namespace perf {

/**
 * @brief Snapshot of counter values at a point in time.
 */
struct CounterSnapshot {
  uint64_t llc_load_misses = 0;   // Last Level Cache load misses (-> DRAM)
  uint64_t llc_loads = 0;         // Total LLC loads
  uint64_t l1d_load_misses = 0;   // L1 data cache load misses
  uint64_t l1d_loads = 0;         // Total L1 data cache loads
  uint64_t dtlb_load_misses = 0;  // Data TLB load misses
  uint64_t dtlb_loads = 0;        // Total data TLB loads
  uint64_t instructions = 0;      // Instructions retired
  uint64_t cycles = 0;            // CPU cycles

  CounterSnapshot operator-(const CounterSnapshot &other) const {
    CounterSnapshot delta;
    delta.llc_load_misses = llc_load_misses - other.llc_load_misses;
    delta.llc_loads = llc_loads - other.llc_loads;
    delta.l1d_load_misses = l1d_load_misses - other.l1d_load_misses;
    delta.l1d_loads = l1d_loads - other.l1d_loads;
    delta.dtlb_load_misses = dtlb_load_misses - other.dtlb_load_misses;
    delta.dtlb_loads = dtlb_loads - other.dtlb_loads;
    delta.instructions = instructions - other.instructions;
    delta.cycles = cycles - other.cycles;
    return delta;
  }

  CounterSnapshot &operator+=(const CounterSnapshot &other) {
    llc_load_misses += other.llc_load_misses;
    llc_loads += other.llc_loads;
    l1d_load_misses += other.l1d_load_misses;
    l1d_loads += other.l1d_loads;
    dtlb_load_misses += other.dtlb_load_misses;
    dtlb_loads += other.dtlb_loads;
    instructions += other.instructions;
    cycles += other.cycles;
    return *this;
  }
};

#ifdef FLATNAV_PERF_COUNTERS

/**
 * @brief Group of hardware performance counters for a single thread.
 *
 * Uses Linux perf_event_open to access hardware PMU counters.
 * Each thread needs its own CounterGroup instance.
 */
class CounterGroup {
public:
  CounterGroup() : _leader_fd(-1), _num_events(0), _started(false) {
    // Event configurations: {type, config}
    // Using PERF_TYPE_HARDWARE and PERF_TYPE_HW_CACHE
    struct EventConfig {
      uint32_t type;
      uint64_t config;
    };

    // Define events we want to measure
    // LLC (Last Level Cache) events use HW_CACHE encoding
    // Format: (cache_id) | (cache_op << 8) | (cache_result << 16)
    constexpr uint64_t LLC_READ_MISS =
        (PERF_COUNT_HW_CACHE_LL) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

    constexpr uint64_t LLC_READ =
        (PERF_COUNT_HW_CACHE_LL) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

    constexpr uint64_t L1D_READ_MISS =
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

    constexpr uint64_t L1D_READ =
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

    constexpr uint64_t DTLB_READ_MISS =
        (PERF_COUNT_HW_CACHE_DTLB) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

    constexpr uint64_t DTLB_READ =
        (PERF_COUNT_HW_CACHE_DTLB) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

    std::vector<EventConfig> events = {
        {PERF_TYPE_HW_CACHE, LLC_READ_MISS},   // 0: LLC load misses
        {PERF_TYPE_HW_CACHE, LLC_READ},        // 1: LLC loads
        {PERF_TYPE_HW_CACHE, L1D_READ_MISS},   // 2: L1D load misses
        {PERF_TYPE_HW_CACHE, L1D_READ},        // 3: L1D loads
        {PERF_TYPE_HW_CACHE, DTLB_READ_MISS},  // 4: DTLB load misses
        {PERF_TYPE_HW_CACHE, DTLB_READ},       // 5: DTLB loads
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},  // 6: instructions
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},    // 7: cycles
    };

    _fds.reserve(events.size());

    for (size_t i = 0; i < events.size(); i++) {
      struct perf_event_attr pe;
      memset(&pe, 0, sizeof(pe));
      pe.size = sizeof(pe);
      pe.type = events[i].type;
      pe.config = events[i].config;
      pe.disabled = 1;
      pe.exclude_kernel = 1;
      pe.exclude_hv = 1;

      // Group leader uses PERF_FORMAT_GROUP for atomic reads
      if (i == 0) {
        pe.read_format = PERF_FORMAT_GROUP;
      }

      // First event is the group leader
      int group_fd = (i == 0) ? -1 : _leader_fd;

      int fd = static_cast<int>(syscall(__NR_perf_event_open, &pe, 0, -1, group_fd, 0));
      if (fd == -1) {
        // Event not available - store -1 and continue
        _fds.push_back(-1);
        if (i == 0) {
          // If leader fails, we can't do anything
          // Only print warning once across all threads
          static std::atomic<bool> warned{false};
          bool expected = false;
          if (warned.compare_exchange_strong(expected, true)) {
            std::cerr << "Warning: perf_event_open failed for leader event.\n"
                      << "To enable hardware counters, run on the HOST (not in container):\n"
                      << "  sudo sysctl -w kernel.perf_event_paranoid=-1\n"
                      << "Then re-run the container with --privileged.\n";
          }
          return;
        }
      } else {
        _fds.push_back(fd);
        _event_map.push_back(static_cast<int>(i));
        _num_events++;
        if (i == 0) {
          _leader_fd = fd;
        }
      }
    }
  }

  ~CounterGroup() {
    for (int fd : _fds) {
      if (fd >= 0) {
        close(fd);
      }
    }
  }

  // Non-copyable
  CounterGroup(const CounterGroup &) = delete;
  CounterGroup &operator=(const CounterGroup &) = delete;

  // Movable
  CounterGroup(CounterGroup &&other) noexcept
      : _fds(std::move(other._fds)),
        _event_map(std::move(other._event_map)),
        _leader_fd(other._leader_fd),
        _num_events(other._num_events),
        _started(other._started) {
    other._leader_fd = -1;
    other._num_events = 0;
    other._started = false;
  }

  /**
   * @brief Start counting (must be called before read()).
   */
  void start() {
    if (_leader_fd >= 0 && !_started) {
      ioctl(_leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
      ioctl(_leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
      _started = true;
    }
  }

  /**
   * @brief Stop counting.
   */
  void stop() {
    if (_leader_fd >= 0 && _started) {
      ioctl(_leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
      _started = false;
    }
  }

  /**
   * @brief Read current counter values atomically via PERF_FORMAT_GROUP.
   *
   * A single read from the group leader fd returns all counter values
   * at the same point in time, avoiding inconsistencies from sequential reads.
   */
  CounterSnapshot read() const {
    CounterSnapshot snap;
    if (_leader_fd < 0 || _num_events == 0)
      return snap;

    // PERF_FORMAT_GROUP layout: { uint64_t nr; uint64_t values[nr]; }
    uint64_t buf[1 + 8]; // nr + up to 8 counter values
    ssize_t expected =
        static_cast<ssize_t>(sizeof(uint64_t) * (1 + _num_events));
    if (::read(_leader_fd, buf, expected) != expected)
      return snap;

    uint64_t nr = buf[0];
    for (size_t i = 0; i < nr && i < _num_events; i++) {
      uint64_t value = buf[1 + i];
      switch (_event_map[i]) {
        case 0: snap.llc_load_misses = value; break;
        case 1: snap.llc_loads = value; break;
        case 2: snap.l1d_load_misses = value; break;
        case 3: snap.l1d_loads = value; break;
        case 4: snap.dtlb_load_misses = value; break;
        case 5: snap.dtlb_loads = value; break;
        case 6: snap.instructions = value; break;
        case 7: snap.cycles = value; break;
      }
    }

    return snap;
  }

  /**
   * @brief Read counters and return delta from a previous snapshot.
   */
  CounterSnapshot readDelta(const CounterSnapshot &previous) const {
    return read() - previous;
  }

  /**
   * @brief Check if counters are available.
   */
  bool available() const { return _leader_fd >= 0; }

private:
  std::vector<int> _fds;
  std::vector<int> _event_map; // maps group index -> event field index
  int _leader_fd;
  size_t _num_events;
  bool _started;
};

#else // !FLATNAV_PERF_COUNTERS

/**
 * @brief No-op implementation when FLATNAV_PERF_COUNTERS is not defined.
 */
class CounterGroup {
public:
  void start() {}
  void stop() {}
  CounterSnapshot read() const { return {}; }
  CounterSnapshot readDelta(const CounterSnapshot &) const { return {}; }
  bool available() const { return false; }
};

#endif // FLATNAV_PERF_COUNTERS

/**
 * @brief Thread-safe accumulator for named counter statistics.
 *
 * Aggregates counter snapshots across multiple measurements and threads.
 */
class CounterStats {
public:
  struct Stats {
    CounterSnapshot total;
    uint64_t call_count = 0;
  };

  /**
   * @brief Accumulate a measurement into a named category.
   */
  void accumulate(const std::string &name, const CounterSnapshot &delta) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto &stats = _stats[name];
    stats.total += delta;
    stats.call_count++;
  }

  /**
   * @brief Get stats for a named category.
   */
  Stats get(const std::string &name) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _stats.find(name);
    if (it != _stats.end()) {
      return it->second;
    }
    return {};
  }

  /**
   * @brief Print all statistics to stdout.
   */
  void print() const {
    std::lock_guard<std::mutex> lock(_mutex);

    std::cout << "\n========== Performance Counter Statistics ==========\n";
    for (const auto &[name, stats] : _stats) {
      std::cout << "\n--- " << name << " ---\n";
      std::cout << "  Calls: " << stats.call_count << "\n";
      if (stats.call_count == 0)
        continue;

      auto &t = stats.total;
      std::cout << "  LLC load misses:  " << t.llc_load_misses;
      if (t.llc_loads > 0) {
        std::cout << " (" << (100.0 * t.llc_load_misses / t.llc_loads) << "% miss rate)";
      }
      std::cout << "\n";

      std::cout << "  L1D load misses:  " << t.l1d_load_misses;
      if (t.l1d_loads > 0) {
        std::cout << " (" << (100.0 * t.l1d_load_misses / t.l1d_loads) << "% miss rate)";
      }
      std::cout << "\n";

      std::cout << "  dTLB load misses: " << t.dtlb_load_misses;
      if (t.dtlb_loads > 0) {
        std::cout << " (" << (100.0 * t.dtlb_load_misses / t.dtlb_loads) << "% miss rate)";
      }
      std::cout << "\n";

      std::cout << "  Instructions:     " << t.instructions << "\n";
      std::cout << "  Cycles:           " << t.cycles << "\n";
      if (t.cycles > 0) {
        std::cout << "  IPC:              " << (1.0 * t.instructions / t.cycles) << "\n";
      }

      // Per-call averages
      std::cout << "  --- Per-call averages ---\n";
      std::cout << "    LLC misses/call:  " << (1.0 * t.llc_load_misses / stats.call_count) << "\n";
      std::cout << "    L1D misses/call:  " << (1.0 * t.l1d_load_misses / stats.call_count) << "\n";
      std::cout << "    dTLB misses/call: " << (1.0 * t.dtlb_load_misses / stats.call_count) << "\n";
      std::cout << "    Cycles/call:      " << (1.0 * t.cycles / stats.call_count) << "\n";
    }
    std::cout << "\n====================================================\n";
  }

  /**
   * @brief Reset all statistics.
   */
  void reset() {
    std::lock_guard<std::mutex> lock(_mutex);
    _stats.clear();
  }

private:
  mutable std::mutex _mutex;
  std::unordered_map<std::string, Stats> _stats;
};

/**
 * @brief Global counter stats instance for easy access.
 */
inline CounterStats &globalStats() {
  static CounterStats instance;
  return instance;
}

/**
 * @brief Thread-local counter group for the current thread.
 *
 * Call initThreadCounters() at the start of each thread that needs
 * to measure performance counters.
 */
inline CounterGroup &threadCounters() {
  thread_local CounterGroup instance;
  return instance;
}

/**
 * @brief Initialize counters for the current thread and start counting.
 *
 * When perf counters are enabled, also pins the thread to a CPU core
 * (round-robin assignment) to eliminate cache migration noise from
 * the scheduler moving threads between cores.
 */
inline void initThreadCounters() {
#ifdef FLATNAV_PERF_COUNTERS
  static std::atomic<int> next_core{0};
  int num_cpus = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
  if (num_cpus > 0) {
    int core = next_core.fetch_add(1) % num_cpus;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
  }
#endif
  threadCounters().start();
}

/**
 * @brief RAII helper for measuring a code section.
 *
 * Usage:
 *   {
 *     ScopedCounter sc("entry_point_selection");
 *     // ... code to measure ...
 *   } // automatically records delta to globalStats()
 */
class ScopedCounter {
public:
  explicit ScopedCounter(const std::string &name)
      : _name(name), _start(threadCounters().read()) {}

  ~ScopedCounter() {
    auto delta = threadCounters().readDelta(_start);
    globalStats().accumulate(_name, delta);
  }

  // Non-copyable, non-movable
  ScopedCounter(const ScopedCounter &) = delete;
  ScopedCounter &operator=(const ScopedCounter &) = delete;

private:
  std::string _name;
  CounterSnapshot _start;
};

} // namespace perf
} // namespace flatnav
