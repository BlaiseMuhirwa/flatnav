/**
 * @file flatnav_config.h
 * @brief Build configuration and platform detection for FlatNav C API
 *
 * This header provides compile-time configuration options, platform
 * detection, and feature flags for different build targets (host vs DPU).
 */

#ifndef FLATNAV_CONFIG_H
#define FLATNAV_CONFIG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Platform Detection
 *============================================================================*/

/* Detect UPMEM DPU target */
#if defined(__dpu__) || defined(FN_FORCE_DPU_TARGET)
#define FN_DPU_TARGET 1
#endif

/* Detect operating system */
#if defined(_WIN32) || defined(_WIN64)
#define FN_PLATFORM_WINDOWS 1
#elif defined(__APPLE__) && defined(__MACH__)
#define FN_PLATFORM_MACOS 1
#elif defined(__linux__)
#define FN_PLATFORM_LINUX 1
#endif

/* Detect compiler */
#if defined(__GNUC__)
#define FN_COMPILER_GCC 1
#define FN_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif
#if defined(__clang__)
#define FN_COMPILER_CLANG 1
#endif
#if defined(_MSC_VER)
#define FN_COMPILER_MSVC 1
#endif

/*============================================================================
 * SIMD Feature Detection (Host Only)
 *============================================================================*/

#ifndef FN_DPU_TARGET

/* Check for SSE support */
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#ifndef FN_DISABLE_SSE
#define FN_USE_SSE 1
#endif
#endif

/* Check for SSE4.1 support */
#if defined(__SSE4_1__)
#ifndef FN_DISABLE_SSE4
#define FN_USE_SSE4 1
#endif
#endif

/* Check for AVX2 support */
#if defined(__AVX2__)
#ifndef FN_DISABLE_AVX2
#define FN_USE_AVX2 1
#endif
#endif

/* Check for AVX-512 support */
#if defined(__AVX512F__) && defined(__AVX512BW__)
#ifndef FN_DISABLE_AVX512
#define FN_USE_AVX512 1
#endif
#endif

/* Check for ARM NEON support */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#ifndef FN_DISABLE_NEON
#define FN_USE_NEON 1
#endif
#endif

#endif /* !FN_DPU_TARGET */

/*============================================================================
 * DPU Constraints
 *============================================================================*/

#ifdef FN_DPU_TARGET
/* DPU has no floating-point unit */
#define FN_NO_FLOAT 1

/* DPU has no threading support */
#define FN_NO_THREADS 1

/* Use static allocation on DPU (limited heap) */
#define FN_STATIC_ALLOC 1

/* Disable all SIMD on DPU */
#undef FN_USE_SSE
#undef FN_USE_SSE4
#undef FN_USE_AVX2
#undef FN_USE_AVX512
#undef FN_USE_NEON

/* Default limits for DPU WRAM */
#ifndef FN_DPU_MAX_NODES
#define FN_DPU_MAX_NODES 1024
#endif

#ifndef FN_DPU_MAX_DIMENSION
#define FN_DPU_MAX_DIMENSION 128
#endif

#ifndef FN_DPU_MAX_EDGES
#define FN_DPU_MAX_EDGES 32
#endif
#endif

/*============================================================================
 * Fixed-Point Configuration (for DPU)
 *============================================================================*/

#ifdef FN_NO_FLOAT
/* Q15.16 fixed-point format */
typedef int32_t fn_fixed32_t;

#define FN_FIXED_SHIFT 16
#define FN_FIXED_ONE (1 << FN_FIXED_SHIFT)
#define FN_FIXED_HALF (1 << (FN_FIXED_SHIFT - 1))

/* Convert float to fixed (host-side preprocessing) */
#define FN_FLOAT_TO_FIXED(x) ((fn_fixed32_t)((x) * FN_FIXED_ONE))

/* Convert fixed to float (for results) */
#define FN_FIXED_TO_FLOAT(x) ((float)(x) / FN_FIXED_ONE)
#endif

/*============================================================================
 * Alignment Macros
 *============================================================================*/

#if defined(FN_COMPILER_GCC) || defined(FN_COMPILER_CLANG)
#define FN_ALIGN(n) __attribute__((aligned(n)))
#elif defined(FN_COMPILER_MSVC)
#define FN_ALIGN(n) __declspec(align(n))
#else
#define FN_ALIGN(n)
#endif

#define FN_ALIGN16 FN_ALIGN(16)
#define FN_ALIGN32 FN_ALIGN(32)
#define FN_ALIGN64 FN_ALIGN(64)

/*============================================================================
 * Inline Hints
 *============================================================================*/

#if defined(FN_COMPILER_GCC) || defined(FN_COMPILER_CLANG)
#define FN_INLINE static inline __attribute__((always_inline))
#define FN_NOINLINE __attribute__((noinline))
#elif defined(FN_COMPILER_MSVC)
#define FN_INLINE static __forceinline
#define FN_NOINLINE __declspec(noinline)
#else
#define FN_INLINE static inline
#define FN_NOINLINE
#endif

/*============================================================================
 * Branch Prediction Hints
 *============================================================================*/

#if defined(FN_COMPILER_GCC) || defined(FN_COMPILER_CLANG)
#define FN_LIKELY(x) __builtin_expect(!!(x), 1)
#define FN_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define FN_LIKELY(x) (x)
#define FN_UNLIKELY(x) (x)
#endif

/*============================================================================
 * Prefetch Hints
 *============================================================================*/

#if defined(FN_COMPILER_GCC) || defined(FN_COMPILER_CLANG)
#define FN_PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#define FN_PREFETCH_W(addr) __builtin_prefetch(addr, 1, 3)
#elif defined(FN_USE_SSE)
#include <xmmintrin.h>
#define FN_PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#define FN_PREFETCH_W(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
#define FN_PREFETCH(addr) ((void)(addr))
#define FN_PREFETCH_W(addr) ((void)(addr))
#endif

/*============================================================================
 * Runtime CPU Feature Detection (Host Only)
 *============================================================================*/

#ifndef FN_DPU_TARGET

/**
 * @brief Check if CPU supports SSE instructions
 * @return 1 if supported, 0 otherwise
 */
int fn_cpu_supports_sse(void);

/**
 * @brief Check if CPU supports SSE4.1 instructions
 * @return 1 if supported, 0 otherwise
 */
int fn_cpu_supports_sse4(void);

/**
 * @brief Check if CPU supports AVX2 instructions
 * @return 1 if supported, 0 otherwise
 */
int fn_cpu_supports_avx2(void);

/**
 * @brief Check if CPU supports AVX-512 instructions
 * @return 1 if supported, 0 otherwise
 */
int fn_cpu_supports_avx512(void);

#endif /* !FN_DPU_TARGET */

#ifdef __cplusplus
}
#endif

#endif /* FLATNAV_CONFIG_H */
