#pragma once

#ifndef NO_SIMD_VECTORIZATION

// _M_AMD64, _M_X64: Code is being compiled for AMD64 or x64 processor.
#if (defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE

#ifdef __AVX__
#define USE_AVX

#ifdef __AVX512F__
#define USE_AVX512
#endif

#endif // __AVX__
#endif
#endif // NO_SIMD_VECTORIZATION

#if defined(USE_AVX) || defined(USE_SSE)

// This would not work on WindowsOS
#include <cpuid.h>
#include <stdint.h>
#include <x86intrin.h>

// void cpuid(int32_t cpu_info[4], int32_t eax, int32_t ecx) {
//   __cpuid_count(eax, ecx, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
// }

// uint64_t xgetbv(unsigned int index) {
//   uint32_t eax, edx;
//   __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
//   return ((uint64_t)edx << 32) | eax;
// }

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

#endif