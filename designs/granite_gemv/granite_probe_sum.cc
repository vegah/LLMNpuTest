// OpenFFLM -- probe_sum entry point. One per translation unit.
// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" {
void probe_sum(const bfloat16 *__restrict full, float *__restrict out,
               unsigned n) {
  aie::accum<accfloat, 32> a = aie::zeros<accfloat, 32>();
  aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
  for (unsigned i = 0; i < n; i += 32)
    a = aie::mac(a, aie::load_v<32>(full + i), one);
  float s = aie::reduce_add(a.template to_vector<float>());
  for (unsigned i = 0; i < 32; ++i) out[i] = s;
}
}
