#pragma once

/**
 * @file PerfCounters.h
 * @brief Lightweight hardware performance counter instrumentation for Linux.
 *
 * This provides inline measurement of cache misses, TLB misses, and other
 * hardware events for specific code sections. Used to validate cache locality
 * hypotheses in entry point selection (anchor vs random initialization).
 *
 * Hardware detection: Uses CPUID at runtime to select the right PMU events
 * for AMD (raw events for L2/L3) vs Intel (generic HW_CACHE interface).
 * See CACHE_BENCHMARKING.md for details on per-vendor event availability.
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
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

namespace flatnav::perf {

// ──────────────────────────────────────────────────────────────────────
// CPU vendor detection (runtime CPUID)
// ──────────────────────────────────────────────────────────────────────
enum class CpuVendor { AMD, INTEL, OTHER };

inline CpuVendor detectCpuVendor() {
#if defined(FLATNAV_PERF_COUNTERS) && (defined(__x86_64__) || defined(__i386__))
  uint32_t eax, ebx, ecx, edx;
  if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
    // "AuthenticAMD"
    if (ebx == 0x68747541 && edx == 0x69746e65 && ecx == 0x444d4163)
      return CpuVendor::AMD;
    // "GenuineIntel"
    if (ebx == 0x756e6547 && edx == 0x49656e69 && ecx == 0x6c65746e)
      return CpuVendor::INTEL;
  }
#endif
  return CpuVendor::OTHER;
}

// ──────────────────────────────────────────────────────────────────────
// Counter snapshot — vendor-neutral field names
// ──────────────────────────────────────────────────────────────────────

/**
 * @brief Snapshot of counter values at a point in time.
 *
 * Field semantics by vendor:
 *   l2_misses   — AMD: l2_cache_misses_from_dc_misses (raw 0x0864)
 *                 Intel: may be 0 (no generic L2 miss counter)
 *   dram_fills  — AMD: l1_data_cache_fills_from_memory (raw 0x4844), L3 miss
 *                 Intel: LLC load misses (= fills from DRAM)
 *   dtlb_*      — Intel only (not enough AMD PMU slots); 0 on AMD.
 */
struct CounterSnapshot {
  uint64_t cycles = 0;            // CPU cycles
  uint64_t instructions = 0;      // Instructions retired
  uint64_t l1d_load_misses = 0;   // L1 data cache load misses
  uint64_t l1d_loads = 0;         // Total L1 data cache loads
  uint64_t l2_misses = 0;         // L2 cache misses (requests that go to L3)
  uint64_t dram_fills = 0;        // Loads filled from DRAM (L3 miss proxy)
  uint64_t dtlb_load_misses = 0;  // Data TLB load misses (Intel only)
  uint64_t dtlb_loads = 0;        // Total data TLB loads (Intel only)

  CounterSnapshot operator-(const CounterSnapshot &other) const {
    CounterSnapshot delta;
    delta.cycles = cycles - other.cycles;
    delta.instructions = instructions - other.instructions;
    delta.l1d_load_misses = l1d_load_misses - other.l1d_load_misses;
    delta.l1d_loads = l1d_loads - other.l1d_loads;
    delta.l2_misses = l2_misses - other.l2_misses;
    delta.dram_fills = dram_fills - other.dram_fills;
    delta.dtlb_load_misses = dtlb_load_misses - other.dtlb_load_misses;
    delta.dtlb_loads = dtlb_loads - other.dtlb_loads;
    return delta;
  }

  CounterSnapshot &operator+=(const CounterSnapshot &other) {
    cycles += other.cycles;
    instructions += other.instructions;
    l1d_load_misses += other.l1d_load_misses;
    l1d_loads += other.l1d_loads;
    l2_misses += other.l2_misses;
    dram_fills += other.dram_fills;
    dtlb_load_misses += other.dtlb_load_misses;
    dtlb_loads += other.dtlb_loads;
    return *this;
  }
};

#ifdef FLATNAV_PERF_COUNTERS

// ──────────────────────────────────────────────────────────────────────
// Event field index — maps array position to CounterSnapshot field
// ──────────────────────────────────────────────────────────────────────
enum EventField : int {
  EF_CYCLES = 0,
  EF_INSTRUCTIONS = 1,
  EF_L1D_LOAD_MISSES = 2,
  EF_L1D_LOADS = 3,
  EF_L2_MISSES = 4,
  EF_DRAM_FILLS = 5,
  EF_DTLB_LOAD_MISSES = 6,
  EF_DTLB_LOADS = 7,
};

/**
 * @brief Group of hardware performance counters for a single thread.
 *
 * Uses Linux perf_event_open to access hardware PMU counters.
 * Each thread needs its own CounterGroup instance.
 *
 * Event selection adapts at runtime based on CPU vendor:
 *   AMD  — 6 events using PERF_TYPE_RAW for L2/L3 (see CACHE_BENCHMARKING.md)
 *   Intel— 8 events using generic PERF_TYPE_HW_CACHE for LLC + dTLB
 */
class CounterGroup {
public:
  CounterGroup() : _leader_fd(-1), _num_events(0), _started(false) {
    struct EventConfig {
      uint32_t type;
      uint64_t config;
      EventField field;
    };

    // ── Generic HW_CACHE encodings (Intel / fallback) ─────────────
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

    // ── AMD raw event encodings ───────────────────────────────────
    // config = (umask << 8) | event_select
    // See CACHE_BENCHMARKING.md §2 for derivation.
    constexpr uint64_t AMD_L2_DC_MISSES = 0x0864;  // l2_cache_misses_from_dc_misses
    constexpr uint64_t AMD_DC_FILLS_DRAM = 0x4844;  // l1_data_cache_fills_from_memory

    // ── Build event list based on detected vendor ─────────────────
    // Cycles and instructions MUST be first — they are universally supported
    // and index 0 becomes the group leader. If the leader fails to open,
    // the entire group is dead.
    _vendor = detectCpuVendor();

    std::vector<EventConfig> events;

    if (_vendor == CpuVendor::AMD) {
      // AMD Zen: 6 GP PMU counters per core.
      // Generic PERF_COUNT_HW_CACHE_LL is unsupported; use raw events.
      events = {
          {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES,   EF_CYCLES},
          {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, EF_INSTRUCTIONS},
          {PERF_TYPE_HW_CACHE, L1D_READ_MISS,              EF_L1D_LOAD_MISSES},
          {PERF_TYPE_HW_CACHE, L1D_READ,                   EF_L1D_LOADS},
          {PERF_TYPE_RAW,      AMD_L2_DC_MISSES,           EF_L2_MISSES},
          {PERF_TYPE_RAW,      AMD_DC_FILLS_DRAM,          EF_DRAM_FILLS},
      };
    } else {
      // Intel / other: 8+ GP PMU counters typically available.
      // Use generic HW_CACHE interface; LLC = L3 on Intel.
      // LLC misses map to dram_fills (= loads that went to DRAM).
      // LLC loads are not directly exposed as a snapshot field but are
      // used to detect whether the counter is active (non-zero → derive
      // miss rate in print/toMap via l2_misses if available).
      events = {
          {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES,   EF_CYCLES},
          {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, EF_INSTRUCTIONS},
          {PERF_TYPE_HW_CACHE, LLC_READ_MISS,              EF_DRAM_FILLS},
          {PERF_TYPE_HW_CACHE, LLC_READ,                   EF_L2_MISSES},
          {PERF_TYPE_HW_CACHE, L1D_READ_MISS,              EF_L1D_LOAD_MISSES},
          {PERF_TYPE_HW_CACHE, L1D_READ,                   EF_L1D_LOADS},
          {PERF_TYPE_HW_CACHE, DTLB_READ_MISS,             EF_DTLB_LOAD_MISSES},
          {PERF_TYPE_HW_CACHE, DTLB_READ,                  EF_DTLB_LOADS},
      };
    }

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

      int fd = static_cast<int>(
          syscall(__NR_perf_event_open, &pe, 0, -1, group_fd, 0));
      if (fd == -1) {
        // Event not available — store -1 and continue
        _fds.push_back(-1);
        if (i == 0) {
          // If leader fails, we can't do anything
          static std::atomic<bool> warned{false};
          bool expected = false;
          if (warned.compare_exchange_strong(expected, true)) {
            std::cerr << "Warning: perf_event_open failed for leader event.\n"
                      << "To enable hardware counters, run on the HOST "
                         "(not in container):\n"
                      << "  sudo sysctl -w kernel.perf_event_paranoid=-1\n"
                      << "Then re-run the container with --privileged.\n";
          }
          return;
        }
      } else {
        _fds.push_back(fd);
        _event_map.push_back(events[i].field);
        _num_events++;
        if (i == 0) {
          _leader_fd = fd;
        }
      }
    }

    // Log which events opened successfully (once per vendor)
    static std::atomic<bool> logged{false};
    bool expected = false;
    if (logged.compare_exchange_strong(expected, true)) {
      const char *vendor_str = (_vendor == CpuVendor::AMD)    ? "AMD"
                               : (_vendor == CpuVendor::INTEL) ? "Intel"
                                                               : "Other";
      std::cerr << "[PerfCounters] CPU vendor: " << vendor_str
                << ", opened " << _num_events << "/" << events.size()
                << " events\n";
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
        _started(other._started),
        _vendor(other._vendor) {
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
        case EF_CYCLES:           snap.cycles = value; break;
        case EF_INSTRUCTIONS:     snap.instructions = value; break;
        case EF_L1D_LOAD_MISSES:  snap.l1d_load_misses = value; break;
        case EF_L1D_LOADS:        snap.l1d_loads = value; break;
        case EF_L2_MISSES:        snap.l2_misses = value; break;
        case EF_DRAM_FILLS:       snap.dram_fills = value; break;
        case EF_DTLB_LOAD_MISSES: snap.dtlb_load_misses = value; break;
        case EF_DTLB_LOADS:       snap.dtlb_loads = value; break;
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

  /**
   * @brief Get the detected CPU vendor.
   */
  CpuVendor vendor() const { return _vendor; }

private:
  std::vector<int> _fds;
  std::vector<EventField> _event_map; // maps group index -> snapshot field
  int _leader_fd;
  size_t _num_events;
  bool _started;
  CpuVendor _vendor;
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
  CpuVendor vendor() const { return CpuVendor::OTHER; }
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
      std::cout << "  L1D load misses:  " << t.l1d_load_misses;
      if (t.l1d_loads > 0) {
        std::cout << " (" << (100.0 * t.l1d_load_misses / t.l1d_loads)
                  << "% miss rate)";
      }
      std::cout << "\n";

      std::cout << "  L2 misses:        " << t.l2_misses << "\n";

      std::cout << "  DRAM fills:       " << t.dram_fills;
      if (t.l2_misses > 0) {
        std::cout << " (" << (100.0 * t.dram_fills / t.l2_misses)
                  << "% of L2 misses went to DRAM)";
      }
      std::cout << "\n";

      std::cout << "  dTLB load misses: " << t.dtlb_load_misses;
      if (t.dtlb_loads > 0) {
        std::cout << " (" << (100.0 * t.dtlb_load_misses / t.dtlb_loads)
                  << "% miss rate)";
      }
      std::cout << "\n";

      std::cout << "  Instructions:     " << t.instructions << "\n";
      std::cout << "  Cycles:           " << t.cycles << "\n";
      if (t.cycles > 0) {
        std::cout << "  IPC:              "
                  << (1.0 * t.instructions / t.cycles) << "\n";
      }

      // Per-call averages
      std::cout << "  --- Per-call averages ---\n";
      std::cout << "    L1D misses/call:  "
                << (1.0 * t.l1d_load_misses / stats.call_count) << "\n";
      std::cout << "    L2 misses/call:   "
                << (1.0 * t.l2_misses / stats.call_count) << "\n";
      std::cout << "    DRAM fills/call:  "
                << (1.0 * t.dram_fills / stats.call_count) << "\n";
      std::cout << "    dTLB misses/call: "
                << (1.0 * t.dtlb_load_misses / stats.call_count) << "\n";
      std::cout << "    Cycles/call:      "
                << (1.0 * t.cycles / stats.call_count) << "\n";
    }
    std::cout << "\n====================================================\n";
  }

  /**
   * @brief Export all statistics as a nested map.
   *
   * Returns a map of { section_name -> { metric_name -> value } }.
   * Automatically convertible to a Python dict by pybind11/stl.h.
   */
  std::unordered_map<std::string, std::unordered_map<std::string, double>>
  toMap() const {
    std::lock_guard<std::mutex> lock(_mutex);
    std::unordered_map<std::string, std::unordered_map<std::string, double>>
        result;

    for (const auto &[name, stats] : _stats) {
      std::unordered_map<std::string, double> m;
      auto &t = stats.total;

      m["call_count"] = static_cast<double>(stats.call_count);

      // Totals
      m["l1d_load_misses"] = static_cast<double>(t.l1d_load_misses);
      m["l1d_loads"] = static_cast<double>(t.l1d_loads);
      m["l2_misses"] = static_cast<double>(t.l2_misses);
      m["dram_fills"] = static_cast<double>(t.dram_fills);
      m["dtlb_load_misses"] = static_cast<double>(t.dtlb_load_misses);
      m["dtlb_loads"] = static_cast<double>(t.dtlb_loads);
      m["instructions"] = static_cast<double>(t.instructions);
      m["cycles"] = static_cast<double>(t.cycles);

      // Rates
      m["l1d_miss_rate"] =
          t.l1d_loads > 0
              ? 100.0 * t.l1d_load_misses / t.l1d_loads
              : 0.0;
      m["dram_fill_rate"] =
          t.l2_misses > 0
              ? 100.0 * t.dram_fills / t.l2_misses
              : 0.0;
      m["dtlb_miss_rate"] =
          t.dtlb_loads > 0
              ? 100.0 * t.dtlb_load_misses / t.dtlb_loads
              : 0.0;
      m["ipc"] =
          t.cycles > 0
              ? 1.0 * t.instructions / t.cycles
              : 0.0;

      // Per-call averages
      if (stats.call_count > 0) {
        double n = static_cast<double>(stats.call_count);
        m["l1d_loads_per_call"] = t.l1d_loads / n;
        m["l1d_misses_per_call"] = t.l1d_load_misses / n;
        m["l2_misses_per_call"] = t.l2_misses / n;
        m["dram_fills_per_call"] = t.dram_fills / n;
        m["dtlb_misses_per_call"] = t.dtlb_load_misses / n;
        m["cycles_per_call"] = t.cycles / n;
        m["instructions_per_call"] = t.instructions / n;
      }

      result[name] = std::move(m);
    }

    return result;
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

} // namespace flatnav::perf
