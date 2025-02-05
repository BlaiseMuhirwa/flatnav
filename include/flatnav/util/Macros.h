#pragma once

#include <atomic>
#include <memory>

#ifndef NO_SIMD_VECTORIZATION

// _M_AMD64, _M_X64: Code is being compiled for AMD64 or x64 processor.
#if (defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE

#ifdef __SSE3__
#define USE_SSE3
#endif  // __SSE3__

#ifdef __SSE4_1__
#define USE_SSE4_1
#endif  // __SSE4_1__

#ifdef __AVX__
#define USE_AVX

#ifdef __AVX512F__

#ifdef __AVX512BW__
#define USE_AVX512BW
#endif  // __AVX512BW__

#ifdef __AVX512VNNI__
#define USE_AVX512VNNI
#endif  // __AVX512VNNI__

#define USE_AVX512
#endif  // __AVX512F__

#endif  // __AVX__
#endif
#endif  // NO_SIMD_VECTORIZATION

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
#endif  // __GNUC__

#define _XCR_XFEATURE_ENABLED_MASK 0

// Cache for AVX and AVX512 support
std::atomic<bool> avx_support_cache{false};
std::atomic<bool> avx_512_support_cache{false};
std::atomic<bool> avx_initialized{false};
std::atomic<bool> avx_512_initialized{false};

/**
 * @brief Initializes the platform support for AVX and AVX512 instructions.
 * This function checks if the CPU and operating system support AVX and AVX512
 * instructions and caches the result for future use.
 *
 * @note This function should be called before using any AVX or AVX512
 * instructions.
 */
void initializePlatformSupport() {
  if (!avx_initialized.load(std::memory_order_acquire)) {
    bool avx_support = false;
    int cpu_info[4];
    cpuid(cpu_info, 0, 0);
    int n_ids = cpu_info[0];

    if (n_ids >= 1) {
      cpuid(cpu_info, 1, 0);
      bool os_uses_xsave_xrstore = (cpu_info[2] & (1 << 27)) != 0;
      bool cpu_avx_support = (cpu_info[2] & (1 << 28)) != 0;
      if (os_uses_xsave_xrstore && cpu_avx_support) {
        uint64_t xcr_feature_mask = xgetbv(0);
        avx_support = (xcr_feature_mask & 0x6) == 0x6;
      }
    }

    avx_support_cache.store(avx_support, std::memory_order_release);
    avx_initialized.store(true, std::memory_order_release);
  }

  if (!avx_512_initialized.load(std::memory_order_acquire)) {
    bool avx512Support = false;
    if (avx_support_cache.load(std::memory_order_acquire)) {
      int cpu_info[4];
      cpuid(cpu_info, 0, 0);
      int n_ids = cpu_info[0];

      if (n_ids >= 0x00000007) {
        cpuid(cpu_info, 0x00000007, 0);
        bool hw_avx512f = (cpu_info[1] & ((int)1 << 16)) != 0;

        if (hw_avx512f) {
          cpuid(cpu_info, 1, 0);
          bool os_uses_xsave_xrstore = (cpu_info[2] & (1 << 27)) != 0;
          bool cpu_avx_support = (cpu_info[2] & (1 << 28)) != 0;

          if (os_uses_xsave_xrstore && cpu_avx_support) {
            uint64_t xcr_feature_mask = xgetbv(0);
            avx512Support = (xcr_feature_mask & 0xe6) == 0xe6;
          }
        }
      }
    }

    avx_512_support_cache.store(avx512Support, std::memory_order_release);
    avx_512_initialized.store(true, std::memory_order_release);
  }
}

bool platformSupportsAvx() {
  if (!avx_initialized.load(std::memory_order_acquire)) {
    initializePlatformSupport();
  }
  return avx_support_cache.load(std::memory_order_acquire);
}

bool platformSupportsAvx512() {
  if (!avx_512_initialized.load(std::memory_order_acquire)) {
    initializePlatformSupport();
  }
  return avx_512_support_cache.load(std::memory_order_acquire);
}

#endif