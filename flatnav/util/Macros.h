#pragma once

#include <atomic>
#include <memory>

#ifndef NO_SIMD_VECTORIZATION

// _M_AMD64, _M_X64: Code is being compiled for AMD64 or x64 processor.
#if (defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE

#ifdef __SSE3__
#define USE_SSE3
#endif // __SSE3__

#ifdef __AVX__
#define USE_AVX

#ifdef __AVX512F__

#ifdef __AVX512BW__
#define USE_AVX512BW
#else
#error "AVX512BW not supported by the compiler"
#endif // __AVX512BW__

// #ifdef __AVX512VNNI__
// #define USE_AVX512VNNI
// #else
// #error "AVX512VNNI not supported by the compiler"
// #endif // __AVX512VNNI__

#define USE_AVX512
#endif // __AVX512F__

#endif // __AVX__
#endif
#endif // NO_SIMD_VECTORIZATION

#if defined(USE_AVX) || defined(USE_SSE)

// This would not work on WindowsOS
#include <cpuid.h>
#include <stdint.h>
#include <x86intrin.h>

void cpuid(int32_t cpu_info[4], int32_t eax, int32_t ecx) {
  __cpuid_count(eax, ecx, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
}

/**
 * @brief Retrieves the value of an extended control register (XCR).
 * This is particularly useful for checking the status of advanced CPU features.
 *
 * @param index The index of the XCR to query. For example, 0 for XCR0, which
 * contains flags for x87 state, SSE state, and AVX state.
 *
 * @return A 64-bit value with the state of the specified XCR. The lower 32 bits
 * are from the EAX register, and the higher 32 bits from the EDX register after
 * the instruction executes.
 *
 * Inline assembly breakdown:
 * - __volatile__ tells the compiler not to optimize this assembly block as its
 * side effects are important.
 * - "xgetbv": The assembly instruction to execute.
 * - "=a"(eax), "=d"(edx): Output operands; after executing 'xgetbv', store EAX
 * in 'eax', and EDX in 'edx'.
 * - "c"(index): Input operand; provides the 'index' parameter to the ECX
 * register before executing 'xgetbv'.
 *
 * The result is constructed by shifting 'edx' left by 32 bits and combining it
 * with 'eax' using bitwise OR.
 */
uint64_t xgetbv(unsigned int index) {
  uint32_t eax, edx;
  __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
  return ((uint64_t)edx << 32) | eax;
}

#ifdef USE_AVX512
#include <immintrin.h>
#endif

#if defined(__GNUC__)
// GCC-specific macros that tells the compiler to align variables on a
// 32/64-byte boundary.
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif // __GNUC__

#define _XCR_XFEATURE_ENABLED_MASK 0

// Cache for AVX and AVX512 support
std::atomic<bool> avxSupportCache{false};
std::atomic<bool> avx512SupportCache{false};
std::atomic<bool> avxInitialized{false};
std::atomic<bool> avx512Initialized{false};

void initializePlatformSupport() {
  if (!avxInitialized.load(std::memory_order_acquire)) {
    bool avxSupport = false;
    int cpu_info[4];
    cpuid(cpu_info, 0, 0);
    int n_ids = cpu_info[0];

    if (n_ids >= 1) {
      cpuid(cpu_info, 1, 0);
      bool osUsesXSAVE_XRSTORE = (cpu_info[2] & (1 << 27)) != 0;
      bool cpuAVXSuport = (cpu_info[2] & (1 << 28)) != 0;
      if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(0);
        avxSupport = (xcrFeatureMask & 0x6) == 0x6;
      }
    }

    avxSupportCache.store(avxSupport, std::memory_order_release);
    avxInitialized.store(true, std::memory_order_release);
  }

  if (!avx512Initialized.load(std::memory_order_acquire)) {
    bool avx512Support = false;
    if (avxSupportCache.load(std::memory_order_acquire)) {
      int cpu_info[4];
      cpuid(cpu_info, 0, 0);
      int n_ids = cpu_info[0];

      if (n_ids >= 0x00000007) {
        cpuid(cpu_info, 0x00000007, 0);
        bool HW_AVX512F = (cpu_info[1] & ((int)1 << 16)) != 0;

        if (HW_AVX512F) {
          cpuid(cpu_info, 1, 0);
          bool osUsesXSAVE_XRSTORE = (cpu_info[2] & (1 << 27)) != 0;
          bool cpuAVXSuport = (cpu_info[2] & (1 << 28)) != 0;

          if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
            uint64_t xcrFeatureMask = xgetbv(0);
            avx512Support = (xcrFeatureMask & 0xe6) == 0xe6;
          }
        }
      }
    }

    avx512SupportCache.store(avx512Support, std::memory_order_release);
    avx512Initialized.store(true, std::memory_order_release);
  }
}

bool platformSupportsAvx() {
  if (!avxInitialized.load(std::memory_order_acquire)) {
    initializePlatformSupport();
  }
  return avxSupportCache.load(std::memory_order_acquire);
}

bool platformSupportsAvx512() {
  if (!avx512Initialized.load(std::memory_order_acquire)) {
    initializePlatformSupport();
  }
  return avx512SupportCache.load(std::memory_order_acquire);
}

#endif