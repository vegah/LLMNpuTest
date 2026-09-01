#pragma once
//===- granite_qrope.h --------------------------------------*- C++ -*-===//
//
// OpenFFLM -- the epilogue that lets q_proj and RoPE share one dispatch.
// SPDX-License-Identifier: Apache-2.0
//
// WHY THIS FUSES WITHOUT AN ALL-GATHER
// ------------------------------------
// Fusing two projections generally needs every core to see the whole
// intermediate vector, so it needs a join to the memtile and a broadcast back.
// RoPE is the exception: it is applied **per head**, and a core that owns whole
// heads can rotate its own slice with no communication at all.
//
// granite is head_dim 64 = 2 tile-rows per head, so any core owning an even
// number of tile-rows owns whole heads. At 8 cores over q_proj's 80 tile-rows
// that is 10 each = 5 heads.
//
// The GEMV accumulates in float32; RoPE consumes bf16. Doing the narrowing here
// rather than in a separate pass means the values never leave L1 between the
// matmul and the rotation.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
#include "granite_elementwise.h"

// A compute tile has only **2 input DMA channels**, and the GEMV already uses
// both (weights and x). Giving cos/sin its own stream is a third and the placer
// refuses: "requires 3 input/1 output DMA channels, but only 2 input/2 output
// available". So cos/sin rides at the end of the x buffer and this kernel finds
// it at a fixed offset -- one stream, no extra channel.
//
// xcs: [GRANITE_K hidden][kHalf cos][kHalf sin], bf16.
// y:   n_heads*64 float32 from the GEMV.  out: n_heads*64 bf16, rotated.
#ifndef GRANITE_QROPE_XOFF
#define GRANITE_QROPE_XOFF 2560
#endif
__attribute__((noinline)) inline void
granite_qrope_impl(const float *__restrict y, const bfloat16 *__restrict xcs,
                   bfloat16 *__restrict out, unsigned n_heads) {
  const bfloat16 *__restrict cs = xcs + GRANITE_QROPE_XOFF;
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  const aie::vector<bfloat16, kHalf> c = aie::load_v<kHalf>(cs);
  const aie::vector<bfloat16, kHalf> s = aie::load_v<kHalf>(cs + kHalf);

  for (unsigned h = 0; h < n_heads; ++h) {
    const float *__restrict yh = y + h * kHeadDim;
    bfloat16 *__restrict oh = out + h * kHeadDim;

    // fp32 -> bf16 for both halves of the head, then the half-split rotation.
    aie::accum<accfloat, kHalf> alo, ahi;
    alo.from_vector(aie::load_v<kHalf>(yh));
    ahi.from_vector(aie::load_v<kHalf>(yh + kHalf));
    aie::vector<bfloat16, kHalf> lo = alo.template to_vector<bfloat16>();
    aie::vector<bfloat16, kHalf> hi = ahi.template to_vector<bfloat16>();

    aie::accum<accfloat, kHalf> lo_c = aie::mul(lo, c);
    aie::accum<accfloat, kHalf> hi_s = aie::mul(hi, s);
    aie::accum<accfloat, kHalf> hi_c = aie::mul(hi, c);
    aie::accum<accfloat, kHalf> lo_s = aie::mul(lo, s);

    aie::store_v(oh, aie::sub(lo_c, hi_s).template to_vector<bfloat16>());
    aie::store_v(oh + kHalf, aie::add(hi_c, lo_s).template to_vector<bfloat16>());
  }
  event1();
}

extern "C" {
void granite_qrope(const float *__restrict y, const bfloat16 *__restrict xcs,
                   bfloat16 *__restrict out, unsigned n_heads) {
  granite_qrope_impl(y, xcs, out, n_heads);
}
}
