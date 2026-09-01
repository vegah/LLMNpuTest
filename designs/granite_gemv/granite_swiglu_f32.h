#pragma once
//===- granite_swiglu_f32.h ---------------------------------*- C++ -*-===//
//
// OpenFFLM -- the epilogue that lets gate_proj, up_proj and SwiGLU share one
// dispatch.  SPDX-License-Identifier: Apache-2.0
//
// WHY THIS FUSES WITHOUT A GATHER
// -------------------------------
// SwiGLU pairs gate[i] with up[i] at the SAME index, so a core that owns the
// same row range of both matrices can combine its own two slices with no
// communication -- the same reason RoPE fuses (granite_qrope.h). Only
// down_proj, which consumes the whole 8192-wide intermediate, needs a real
// gather.
//
// Both inputs arrive as the GEMV's float32 accumulators and never leave L1;
// the narrowing to bf16 happens here, once, on the result.
//
//     h[i] = silu(gate[i]) * up[i],   silu(x) = x * (tanh(x/2) + 1) / 2
//
// `tanh(x/2) = 2*sigmoid(x) - 1` is an identity, so the sigmoid is exact; only
// aie::tanh's own implementation approximates.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

__attribute__((noinline)) inline void
granite_swiglu_f32_impl(const float *__restrict gate, const float *__restrict up,
                        bfloat16 *__restrict out, unsigned n) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned V = 32;
  const aie::vector<bfloat16, V> half = aie::broadcast<bfloat16, V>((bfloat16)0.5f);
  const aie::vector<bfloat16, V> one = aie::broadcast<bfloat16, V>((bfloat16)1.0f);

  for (unsigned i = 0; i < n; i += V) {
    aie::accum<accfloat, V> ga, ua;
    ga.from_vector(aie::load_v<V>(gate + i));
    ua.from_vector(aie::load_v<V>(up + i));
    aie::vector<bfloat16, V> g = ga.template to_vector<bfloat16>();
    aie::vector<bfloat16, V> u = ua.template to_vector<bfloat16>();

    aie::accum<accfloat, V> gh = aie::mul(g, half);
    aie::vector<bfloat16, V> t = aie::tanh<bfloat16>(gh.template to_vector<float>());
    aie::vector<bfloat16, V> sig =
        aie::mul(aie::add(t, one), half).template to_vector<bfloat16>();
    aie::vector<bfloat16, V> silu = aie::mul(g, sig).template to_vector<bfloat16>();
    aie::store_v(out + i, aie::mul(silu, u).template to_vector<bfloat16>());
  }
  event1();
}

extern "C" {
void granite_swiglu_f32(const float *__restrict gate, const float *__restrict up,
                        bfloat16 *__restrict out, unsigned n) {
  granite_swiglu_f32_impl(gate, up, out, n);
}
}
