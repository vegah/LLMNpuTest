// OpenFFLM -- probe_fill entry point. One per translation unit.
// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" {
void probe_fill(bfloat16 *__restrict out, unsigned n, unsigned tag) {
  aie::vector<bfloat16, 32> v = aie::broadcast<bfloat16, 32>((bfloat16)(float)tag);
  for (unsigned i = 0; i < n; i += 32) aie::store_v(out + i, v);
}
}
