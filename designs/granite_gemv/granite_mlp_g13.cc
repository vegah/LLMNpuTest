// OpenFFLM -- MLP GEMV group 13, 2 K-tiles, writing at a row offset.
// One entry point per translation unit.
// SPDX-License-Identifier: Apache-2.0
#define GRANITE_TILES_PER_CALL 2
#define GRANITE_BATCH 1
#define GRANITE_K 2560
#include "granite_gemv.h"

extern "C" {
void granite_mlp_g13(const uint8_t *__restrict t, const bfloat16 *__restrict x,
                     float *__restrict y, unsigned row) {
  gemv_q4_group(t, x, 13, y + row * kRows);
}
}
