/**
 * @file simd_dispatch.c
 * @brief Runtime SIMD function dispatch initialization
 */

#include <flatnav/flatnav.h>

/* Forward declarations for dispatch initialization functions */
extern void fn__init_l2_dispatch(void);
extern void fn__init_ip_dispatch(void);

/**
 * @brief Initialize all SIMD dispatch tables
 *
 * Called during library initialization to select the best SIMD
 * implementations based on CPU features.
 */
void fn__init_simd_dispatch(void) {
#ifndef FN_DPU_TARGET
  fn__init_l2_dispatch();
  fn__init_ip_dispatch();
#endif
}
