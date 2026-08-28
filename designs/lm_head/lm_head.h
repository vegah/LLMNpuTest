#pragma once
//===- lm_head.h  -------------------------------------------*- C++ -*-===//
//
// OpenFFLM -- W8A16 GEMV for the lm_head projection, on the AIE core.
// SPDX-License-Identifier: Apache-2.0
//
// y[248320] = W[248320, 1024] @ x[1024], W in q4nx's q8 form.
//
// WHY THIS ONE FIRST
// ------------------
// It is 254.3M of the model's 752M MACs per token -- 34% -- in a single
// projection, which gives it by far the best work-per-dispatch ratio in the
// model. That matters more than the arithmetic: TileFuse measured that decode
// on XDNA2 stays iGPU-dominated because millisecond-scale dispatch overhead
// swamps microsecond-scale GEMV, and NpuEmbeddings tasks/0024 measured the same
// wall at ~55 us + ~286 us per column. lm_head is the one projection large
// enough that dispatch cannot dominate it.
//
// WHY IT IS CHEAP, WHICH IS A PROPERTY OF THE LAYOUT
// --------------------------------------------------
// One q8 tile is 32 output rows x 256 K:
//
//   scales[256] bf16   at tile[0 : 512]      index = kb*32 + rb*16 + r
//   codes [8192] int8  at tile[512 : 8704]   index = rb*4096 + k*16 + r
//
// and the row within the tile is rb*16 + r. So `load_v<32>(s + kb*32)` is
// exactly the 32 output rows in order, and concatenating two 16-lane code loads
// 4096 apart gives those same 32 rows at one k. No gather, no shuffle, one
// 32-lane MAC per activation element. FLM built the layout for this.
//
// The scale is constant across the 32 K of a block, so it factors out of the
// inner sum -- one multiply per block, not one per element.
//
// This kernel is bandwidth-bound, not compute-bound: it consumes one weight
// byte per MAC, and a core can issue far more MACs per cycle than the shim can
// deliver bytes. So the inner loop is written for clarity. The number to beat
// is 254 MB / 45.5 GB/s (the roofline tasks/0128 measured) = 5.6 ms per token.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned kRows = 32;        // output rows per tile
static constexpr unsigned kKBlocks = 8;      // 32-wide K blocks per tile
static constexpr unsigned kKInBlock = 32;
static constexpr unsigned kTileK = 256;      // K per tile
static constexpr unsigned kScaleBytes = 512;
static constexpr unsigned kRowBlockStride = 4096;  // in codes

// KT selects which 256-wide slice of x this tile covers; FIRST starts the
// accumulator rather than adding to it, so the four calls chain without needing
// a separate zeroing pass over the output.
template <unsigned KT, bool FIRST>
static inline void gemv_q8_tile(const uint8_t *__restrict tile,
                                const bfloat16 *__restrict x,
                                float *__restrict y) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *__restrict s = (const bfloat16 *)tile;
  const int8_t *__restrict c = (const int8_t *)(tile + kScaleBytes);
  const bfloat16 *__restrict xt = x + KT * kTileK;

  // One fp32 accumulator for the whole tile (trap 2). FIRST starts it, the
  // other three K tiles pick up where the previous one left off, so the four
  // calls chain without a separate zeroing pass over the output.
  aie::accum<accfloat, kRows> acc;
  if constexpr (FIRST)
    acc = aie::zeros<accfloat, kRows>();
  else
    acc.from_vector(aie::load_v<kRows>(y));

  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    // The 32 scales for this K block are the 32 output rows in order.
    aie::vector<bfloat16, kRows> sv = aie::load_v<kRows>(s + kb * kRows);

    // int8 codes are exact in bf16, so this partial sum is exact in fp32.
    aie::accum<accfloat, kRows> part = aie::zeros<accfloat, kRows>();
    for (unsigned kk = 0; kk < kKInBlock; ++kk) {
      const unsigned k = kb * kKInBlock + kk;
      aie::vector<int8_t, 16> c0 = aie::load_v<16>(c + k * 16);
      aie::vector<int8_t, 16> c1 = aie::load_v<16>(c + kRowBlockStride + k * 16);
      part = aie::mac(part,
                      aie::to_float<bfloat16>(aie::concat(c0, c1), 0), xt[k]);
    }

    // Apply the block scale. The obvious `part * scale` needs an fp32 x fp32
    // vector multiply, and AIE2P has no fp32 vector multiplier -- aie_api
    // compiles it and the result is zero, in silence. Rounding the partial to
    // bf16 instead would cost ~2^-9 of it.
    //
    // So split the fp32 partial into two bf16 halves and scale both. Each
    // product is bf16 x bf16, which is native, and 8 + 8 mantissa bits land
    // exactly in the fp32 accumulator -- about 2^-17 overall. Eight times per
    // tile against 256 MACs, on a kernel that is bandwidth-bound: free.
    aie::vector<bfloat16, kRows> hi = part.template to_vector<bfloat16>();
    aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).to_vector<bfloat16>();
    acc = aie::mac(acc, hi, sv);
    acc = aie::mac(acc, lo, sv);
  }

  aie::store_v(y, acc.template to_vector<float>());
  event1();
}

extern "C" {

// One entry point per K tile, so the tile index is a compile-time constant and
// the design needs no runtime parameter for it. They live in four separate
// translation units because IRON compiles the kernel source once per
// ExternalFunction -- four functions from one .cc means four objects that each
// define all four symbols, and the link fails on duplicates.
#define LMHEAD_ENTRY(N, FIRST)                                                   void lmhead_q8_k##N(const uint8_t *__restrict t,                                                   const bfloat16 *__restrict x, float *__restrict y) {         gemv_q8_tile<N, FIRST>(t, x, y);                                             }

}  // extern "C"
