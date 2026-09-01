// OpenFFLM -- CPU baseline for the granite q4nx q4 GEMV, for comparison against
// the NPU design in this directory.
// SPDX-License-Identifier: Apache-2.0
//
// Same file, same bytes, same arithmetic as granite_gemv.h -- so the comparison
// is like-for-like rather than against a straw man. Two numbers are reported:
//
//   1. STREAM   -- how fast this machine can simply *read* the weight bytes.
//                  Decode is memory-bound, so this is the floor no CPU GEMV
//                  implementation can beat, however well written. If the NPU
//                  loses to this, it loses to any competent CPU engine.
//   2. GEMV     -- the actual W4A16 GEMV, AVX2 + FMA, one thread per core.
//
// Weights are read from the user's installed model; nothing is redistributed.
//
//   cl /O2 /arch:AVX2 /std:c++17 /EHsc cpu_baseline.cpp
//   cpu_baseline.exe <model.q4nx> <byte_offset> <n_tile_rows> <k_tiles> [threads]

#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

static constexpr int kTileBytes = 5120;
static constexpr int kRows = 32;      // output rows per tile
static constexpr int kKBlocks = 8;    // 32-wide K blocks per tile
static constexpr int kTileK = 256;

static inline __m256 bf16x8_to_f32(const uint16_t *p) {
  // bf16 -> f32 is a 16-bit left shift; no table, no conversion instruction.
  __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

// y[32] for one tile-row: sum over K tiles of one 5120-byte tile.
static void gemv_tile_row(const uint8_t *row, const float *x, int k_tiles,
                          float *y) {
  __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                   _mm256_setzero_ps(), _mm256_setzero_ps()};
  for (int kt = 0; kt < k_tiles; ++kt) {
    const uint8_t *tile = row + kt * kTileBytes;
    const uint16_t *dp = reinterpret_cast<const uint16_t *>(tile);
    const uint16_t *mp = reinterpret_cast<const uint16_t *>(tile + 512);
    const uint8_t *nib = tile + 1024;
    const float *xt = x + kt * kTileK;

    for (int kb = 0; kb < kKBlocks; ++kb) {
      // sum of x over this K block -- the whole cost of the Q4_1 minimum.
      __m256 xs8 = _mm256_setzero_ps();
      for (int t = 0; t < 32; t += 8)
        xs8 = _mm256_add_ps(xs8, _mm256_loadu_ps(xt + kb * 32 + t));
      __m128 lo = _mm256_castps256_ps128(xs8);
      __m128 hi = _mm256_extractf128_ps(xs8, 1);
      lo = _mm_add_ps(lo, hi);
      lo = _mm_hadd_ps(lo, lo);
      lo = _mm_hadd_ps(lo, lo);
      const __m256 xs = _mm256_broadcastss_ps(lo);

      for (int rb = 0; rb < 2; ++rb) {
        const uint8_t *src = nib + rb * 2048 + kb * 256;
        __m256 p0 = _mm256_setzero_ps(), p1 = _mm256_setzero_ps();

        for (int kk = 0; kk < 32; ++kk) {
          // 8 bytes = 16 nibbles = 16 rows at one k. Low nibble is the even
          // row, so unpacklo restores row order: lo0, hi0, lo1, hi1, ...
          __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + kk * 8));
          __m128i mask = _mm_set1_epi8(0x0F);
          __m128i nlo = _mm_and_si128(b, mask);
          __m128i nhi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
          __m128i codes = _mm_unpacklo_epi8(nlo, nhi);

          __m256 xv = _mm256_set1_ps(xt[kb * 32 + kk]);
          __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(codes));
          __m256 f1 = _mm256_cvtepi32_ps(
              _mm256_cvtepu8_epi32(_mm_srli_si128(codes, 8)));
          p0 = _mm256_fmadd_ps(f0, xv, p0);
          p1 = _mm256_fmadd_ps(f1, xv, p1);
        }

        const int g = kb * kRows + rb * 16;
        __m256 d0 = bf16x8_to_f32(dp + g), d1 = bf16x8_to_f32(dp + g + 8);
        __m256 m0 = bf16x8_to_f32(mp + g), m1 = bf16x8_to_f32(mp + g + 8);
        int a = rb * 2;
        acc[a] = _mm256_fmadd_ps(m0, xs, _mm256_fmadd_ps(d0, p0, acc[a]));
        acc[a + 1] = _mm256_fmadd_ps(m1, xs, _mm256_fmadd_ps(d1, p1, acc[a + 1]));
      }
    }
  }
  for (int i = 0; i < 4; ++i) _mm256_storeu_ps(y + i * 8, acc[i]);
}

template <typename F>
static double timed(int iters, F &&f) {
  using clk = std::chrono::steady_clock;
  f();  // warm
  auto t0 = clk::now();
  for (int i = 0; i < iters; ++i) f();
  auto t1 = clk::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <model.q4nx> <offset> <n_tile_rows> <k_tiles> "
                 "[threads]\n",
                 argv[0]);
    return 2;
  }
  const char *path = argv[1];
  const long long off = atoll(argv[2]);
  const int tile_rows = atoi(argv[3]);
  const int k_tiles = atoi(argv[4]);
  const int nthreads = argc > 5 ? atoi(argv[5])
                                : (int)std::thread::hardware_concurrency();
  const long long row_bytes = (long long)k_tiles * kTileBytes;
  const long long total = (long long)tile_rows * row_bytes;
  const int K = k_tiles * kTileK;

  std::vector<uint8_t> w((size_t)total);
  {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 1; }
    f.seekg(off);
    f.read(reinterpret_cast<char *>(w.data()), total);
    if (f.gcount() != total) {
      std::fprintf(stderr, "short read: %lld of %lld\n", (long long)f.gcount(), total);
      return 1;
    }
  }
  std::vector<float> x((size_t)K), y((size_t)tile_rows * kRows);
  for (int i = 0; i < K; ++i) x[i] = (float)((i * 37 % 101) - 50) / 50.0f;

  auto parallel = [&](auto &&body) {
    std::vector<std::thread> ts;
    std::atomic<int> next{0};
    for (int t = 0; t < nthreads; ++t)
      ts.emplace_back([&] {
        for (int r = next++; r < tile_rows; r = next++) body(r);
      });
    for (auto &t : ts) t.join();
  };

  // 1. STREAM: the memory-bandwidth floor. Reads every weight byte and does
  //    almost nothing with them, so it bounds any CPU GEMV from below.
  std::atomic<uint64_t> sink{0};
  double ms_stream = timed(3, [&] {
    parallel([&](int r) {
      const uint8_t *p = w.data() + (size_t)r * row_bytes;
      __m256i s = _mm256_setzero_si256();
      for (long long i = 0; i + 32 <= row_bytes; i += 32)
        s = _mm256_add_epi8(s, _mm256_loadu_si256((const __m256i *)(p + i)));
      sink += (uint64_t)_mm256_extract_epi8(s, 0);
    });
  });

  // 2. GEMV: the real thing.
  double ms_gemv = timed(3, [&] {
    parallel([&](int r) {
      gemv_tile_row(w.data() + (size_t)r * row_bytes, x.data(), k_tiles,
                    y.data() + (size_t)r * kRows);
    });
  });

  const double mb = total / 1e6;
  std::printf("threads      %d\n", nthreads);
  std::printf("weights      %.1f MB  (%d tile-rows x %d K-tiles, N=%d K=%d)\n",
              mb, tile_rows, k_tiles, tile_rows * kRows, K);
  std::printf("STREAM       %7.2f ms   %5.1f GB/s   <- CPU memory floor\n",
              ms_stream, mb / ms_stream);
  std::printf("GEMV         %7.2f ms   %5.1f GB/s   %.1f GFLOP/s\n", ms_gemv,
              mb / ms_gemv, 2.0 * tile_rows * kRows * K / ms_gemv / 1e6);
  std::printf("y[0..3]      %.8f %.8f %.8f %.8f\n", y[0], y[1], y[2], y[3]);
  std::printf("checksum     %llu\n", (unsigned long long)sink.load());
  return 0;
}
