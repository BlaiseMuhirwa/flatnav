/**
 * @file flatnav.c
 * @brief Library initialization and utility functions
 */

#include <flatnav/flatnav.h>
#include <stdio.h>
#include <string.h>

/*============================================================================
 * Global State
 *============================================================================*/

static int g_initialized = 0;
static char g_version_str[32] = {0};
static char g_build_info[256] = {0};

/* Forward declarations for SIMD dispatch initialization */
#ifndef FN_DPU_TARGET
extern void fn__init_simd_dispatch(void);
#endif

/*============================================================================
 * Library Initialization
 *============================================================================*/

FN_API void fn_init(void) {
  if (g_initialized) {
    return;
  }

  /* Build version string */
  snprintf(g_version_str, sizeof(g_version_str), "%d.%d.%d", FN_VERSION_MAJOR, FN_VERSION_MINOR,
           FN_VERSION_PATCH);

  /* Build info string */
  snprintf(g_build_info, sizeof(g_build_info),
           "FlatNav C API v%s ("
#ifdef FN_DPU_TARGET
           "DPU"
#else
           "Host"
#endif
#ifdef FN_USE_AVX512
           ", AVX-512"
#endif
#ifdef FN_USE_AVX2
           ", AVX2"
#endif
#ifdef FN_USE_SSE4
           ", SSE4.1"
#endif
#ifdef FN_USE_SSE
           ", SSE"
#endif
#ifdef FN_USE_NEON
           ", NEON"
#endif
#ifndef FN_NO_THREADS
           ", Threads"
#endif
           ")",
           g_version_str);

#ifndef FN_DPU_TARGET
  /* Initialize SIMD dispatch */
  fn__init_simd_dispatch();
#endif

  g_initialized = 1;
}

FN_API void fn_cleanup(void) {
  g_initialized = 0;
}

FN_API int fn_is_initialized(void) {
  return g_initialized;
}

/*============================================================================
 * Version Information
 *============================================================================*/

FN_API const char* fn_version_string(void) {
  if (g_version_str[0] == '\0') {
    snprintf(g_version_str, sizeof(g_version_str), "%d.%d.%d", FN_VERSION_MAJOR, FN_VERSION_MINOR,
             FN_VERSION_PATCH);
  }
  return g_version_str;
}

FN_API int fn_version_major(void) {
  return FN_VERSION_MAJOR;
}

FN_API int fn_version_minor(void) {
  return FN_VERSION_MINOR;
}

FN_API int fn_version_patch(void) {
  return FN_VERSION_PATCH;
}

/*============================================================================
 * Build Information
 *============================================================================*/

FN_API const char* fn_build_info(void) {
  if (g_build_info[0] == '\0') {
    fn_init(); /* Ensure build info is populated */
  }
  return g_build_info;
}

FN_API int fn_is_dpu_build(void) {
#ifdef FN_DPU_TARGET
  return 1;
#else
  return 0;
#endif
}

FN_API int fn_has_simd(void) {
#if defined(FN_USE_AVX512) || defined(FN_USE_AVX2) || defined(FN_USE_SSE4) || defined(FN_USE_SSE) || \
    defined(FN_USE_NEON)
  return 1;
#else
  return 0;
#endif
}

/*============================================================================
 * Error Handling
 *============================================================================*/

FN_API const char* fn_error_string(fn_error_t error) {
  switch (error) {
    case FN_ERR_OK:
      return "Success";
    case FN_ERR_NULL_PTR:
      return "NULL pointer";
    case FN_ERR_INVALID_ARG:
      return "Invalid argument";
    case FN_ERR_OUT_OF_MEMORY:
      return "Out of memory";
    case FN_ERR_INDEX_FULL:
      return "Index is full";
    case FN_ERR_IO_ERROR:
      return "I/O error";
    case FN_ERR_INVALID_FORMAT:
      return "Invalid format";
    case FN_ERR_DIMENSION_MISMATCH:
      return "Dimension mismatch";
    case FN_ERR_NOT_SUPPORTED:
      return "Not supported";
    case FN_ERR_NOT_INITIALIZED:
      return "Not initialized";
    case FN_ERR_INTERNAL:
      return "Internal error";
    default:
      return "Unknown error";
  }
}

/*============================================================================
 * Data Type Utilities
 *============================================================================*/

FN_API size_t fn_data_type_size(fn_data_type_t dtype) {
  switch (dtype) {
    case FN_DATA_FLOAT32:
      return sizeof(float);
    case FN_DATA_INT8:
      return sizeof(int8_t);
    case FN_DATA_UINT8:
      return sizeof(uint8_t);
    case FN_DATA_INT16:
      return sizeof(int16_t);
    case FN_DATA_INT32:
      return sizeof(int32_t);
    default:
      return 0;
  }
}

FN_API const char* fn_data_type_name(fn_data_type_t dtype) {
  switch (dtype) {
    case FN_DATA_FLOAT32:
      return "float32";
    case FN_DATA_INT8:
      return "int8";
    case FN_DATA_UINT8:
      return "uint8";
    case FN_DATA_INT16:
      return "int16";
    case FN_DATA_INT32:
      return "int32";
    default:
      return "undefined";
  }
}

FN_API const char* fn_metric_type_name(fn_metric_type_t metric) {
  switch (metric) {
    case FN_METRIC_L2:
      return "l2";
    case FN_METRIC_IP:
      return "ip";
    default:
      return "unknown";
  }
}

/*============================================================================
 * CPU Feature Detection (Host Only)
 *============================================================================*/

#ifndef FN_DPU_TARGET

#if defined(FN_COMPILER_GCC) || defined(FN_COMPILER_CLANG)
#include <cpuid.h>

static void fn__cpuid(int info[4], int leaf) {
  __cpuid(leaf, info[0], info[1], info[2], info[3]);
}

static void fn__cpuidex(int info[4], int leaf, int subleaf) {
  __cpuid_count(leaf, subleaf, info[0], info[1], info[2], info[3]);
}

#elif defined(FN_COMPILER_MSVC)
#include <intrin.h>

static void fn__cpuid(int info[4], int leaf) {
  __cpuid(info, leaf);
}

static void fn__cpuidex(int info[4], int leaf, int subleaf) {
  __cpuidex(info, leaf, subleaf);
}

#else
/* Fallback: no CPUID support */
static void fn__cpuid(int info[4], int leaf) {
  (void)leaf;
  info[0] = info[1] = info[2] = info[3] = 0;
}

static void fn__cpuidex(int info[4], int leaf, int subleaf) {
  (void)leaf;
  (void)subleaf;
  info[0] = info[1] = info[2] = info[3] = 0;
}
#endif

int fn_cpu_supports_sse(void) {
  int info[4];
  fn__cpuid(info, 1);
  return (info[3] & (1 << 25)) != 0; /* EDX bit 25 */
}

int fn_cpu_supports_sse4(void) {
  int info[4];
  fn__cpuid(info, 1);
  return (info[2] & (1 << 19)) != 0; /* ECX bit 19 */
}

int fn_cpu_supports_avx2(void) {
  int info[4];
  fn__cpuidex(info, 7, 0);
  return (info[1] & (1 << 5)) != 0; /* EBX bit 5 */
}

int fn_cpu_supports_avx512(void) {
  int info[4];
  fn__cpuidex(info, 7, 0);
  /* Check for AVX-512F (bit 16) and AVX-512BW (bit 30) */
  return ((info[1] & (1 << 16)) != 0) && ((info[1] & (1 << 30)) != 0);
}

#endif /* !FN_DPU_TARGET */
