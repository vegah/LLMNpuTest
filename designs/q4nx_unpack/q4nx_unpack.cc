//===- q4nx_unpack.cc ---------------------------------------*- C++ -*-===//
//
// OpenFFLM PoC -- 4-bit weights unpacked to bf16 on the AIE core.
// SPDX-License-Identifier: Apache-2.0
//
// This is the piece NpuEmbeddings deferred: AIE2P has no int4 MAC, so int4 is a
// storage format and somebody has to dequantise it. Doing that on the core is
// the whole reason an open q4nx GEMM is possible at all -- if the host had to
// widen the weights first, the NPU would be reading bf16 from DDR and the 4-bit
// format would buy nothing.
//
// One q4nx tile is 32 output rows x 256 K, stored as two row-blocks of 16:
//
//   meta[0   .. 255]  bf16 d     index = kb*32 + rb*16 + r    (kb = k / 32)
//   meta[256 .. 511]  bf16 m     same index
//   nib[0 .. 4095]    4096 bytes, weight i at byte i>>1, low nibble when even
//   weight index i  = rb*4096 + k*16 + r        (k = 0..255, r = 0..15)
//
//   w = code * d + m            GGUF Q4_1 semantics: scale and minimum.
//
// The layout is what makes this cheap. Because the row index is the FASTEST
// axis, 16 consecutive weights are 16 different rows at one k -- so they need
// 16 different scales, and those 16 scales are constant for a whole k-block.
// Load them once per (rb, kb), splat to 64 lanes, and the inner loop is a mask,
// a shift, a zip and one mul-add. No gather, no per-group reload.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned kRowBlocks = 2;
static constexpr unsigned kRows = 16;   // rows per row-block
static constexpr unsigned kKBlocks = 8; // k-blocks per tile (256 K / 32)
static constexpr unsigned kKInBlock = 32;
static constexpr unsigned kMeta = 256; // d entries, and m entries

extern "C" {

// nib: 4096 B packed. meta: 512 bf16, d then m. out: 8192 bf16.
void q4nx_unpack_tile(const uint8_t *__restrict nib,
                      const bfloat16 *__restrict meta,
                      bfloat16 *__restrict out) {
  event0();

  // The AIE default is `floor`, a systematic downward bias rather than
  // symmetric noise, and it would be baked into every weight (trap 2b).
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *__restrict dp = meta;
  const bfloat16 *__restrict mp = meta + kMeta;

  for (unsigned rb = 0; rb < kRowBlocks; ++rb)
    for (unsigned kb = 0; kb < kKBlocks; ++kb) {
      const unsigned g = kb * (kRowBlocks * kRows) + rb * kRows;

      // 16 scales, one per row, constant across this whole k-block. Splat to 64
      // lanes so they line up with 4 k's worth of unpacked codes.
      aie::vector<bfloat16, kRows> d16 = aie::load_v<kRows>(dp + g);
      aie::vector<bfloat16, kRows> m16 = aie::load_v<kRows>(mp + g);
      aie::vector<bfloat16, 64> d64 = aie::concat(d16, d16, d16, d16);
      aie::vector<bfloat16, 64> m64 = aie::concat(m16, m16, m16, m16);

      const uint8_t *__restrict src = nib + rb * 2048 + kb * 256;
      bfloat16 *__restrict dst = out + rb * 4096 + kb * (kKInBlock * kRows);

      // 64 bytes = 128 weights = 8 k's, per iteration.
      for (unsigned kk = 0; kk < kKInBlock; kk += 8) {
        aie::vector<uint8_t, 64> p = aie::load_v<64>(src + kk * 8);

        // uint8 -> bf16 is one operation on Gen2. The high nibble is masked
        // rather than shifted, and `to_float`'s shift argument -- the position
        // of the binary point -- does the divide by 16 for free. aie::downshift
        // on uint8 would work too, but it lowers to srs_to_v64uint8, which
        // Peano marks deprecated and the build promotes to an error.
        aie::vector<bfloat16, 64> flo =
            aie::to_float<bfloat16>(aie::bit_and((uint8_t)0x0F, p), 0);
        aie::vector<bfloat16, 64> fhi =
            aie::to_float<bfloat16>(aie::bit_and((uint8_t)0xF0, p), 4);

        // Low nibble is the even weight index, so zipping lo with hi at chunk
        // size 1 restores weight order: [lo0, hi0, lo1, hi1, ...].
        auto [c0, c1] = aie::interleave_zip(flo, fhi, 1);

        aie::store_v(dst + kk * kRows,
                     aie::add(aie::mul(c0, d64), m64).to_vector<bfloat16>());
        aie::store_v(dst + kk * kRows + 64,
                     aie::add(aie::mul(c1, d64), m64).to_vector<bfloat16>());
      }
    }

  event1();
}

}  // extern "C"
