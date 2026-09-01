// OpenFFLM -- debug entry point: copy the raw attention state out unchanged.
// One entry point per translation unit; see granite_attention.h.
// SPDX-License-Identifier: Apache-2.0
//
// Emits acc[64] | m | l instead of the normalised output, so the running max,
// the normaliser and the unnormalised accumulator can each be compared against
// numpy separately. Guessing which of the three was wrong cost three build
// cycles and two wrong answers; this settles it in one.
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" {
void granite_attn_dump(const float *__restrict state, float *__restrict out) {
  for (unsigned i = 0; i < 66; ++i) out[i] = state[i];
}
}
