r"""What does it cost to issue a small matmul, against what it computes?

This is the measurement that decides whether OpenFFLM is a project or a demo.

lm_head answered the kernel question and dodged the dispatch one: it is 270 MB in
a single dispatch, 7166 us of work, so any plausible issue cost is noise. The
other 186 matmuls in a token are the opposite -- median 2.29 MB, about 47 us of
work each. NpuEmbeddings tasks/0024 measured a design switch at ~55 us + ~286 us
per column, which is more than one of those matmuls costs to run. If that is what
issuing one costs, the NPU loses to the CPU on everything except lm_head.

So: sweep the same design over sizes, fit t = fixed + bytes / bandwidth, and read
the intercept. Repeated invocation of ONE loaded design is the FLOOR on
per-dispatch cost -- the real runtime alternates between about ten shapes, which
costs this or more depending on whether they share an xclbin. A floor is enough
to decide: if even this is ~300 us, nothing else matters.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\lm_head\dispatch_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron.device import from_name
from aie.utils.benchmark import run_iters

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "tools"))
import q4nx  # noqa: E402
from lm_head import K, ROW_BYTES, ROWS_PER_TILE, lm_head  # noqa: E402

CORES = 8
# Tile-row counts, all divisible by CORES. 64 tile-rows is 2.23 MB, the median
# layer matmul; 7760 is lm_head itself, at the other end by a factor of 121.
SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 7760]
MEDIAN_LAYER_MB = 2.29
N_LAYER_MATMULS = 186
N_FUSED_GROUPS = 97  # ~4 per layer + lm_head, the most that can be fused given
                     # that norms, RoPE, attention and the delta rule stay on CPU


def main() -> int:
    f = q4nx.Q4NX(Path.home() / ".cache" / "openfflm" / "Qwen3.5-0.8B-NPU2" / "model.q4nx")
    first, _ = f.header["lm_head.weight"]["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + first)
        blob = np.frombuffer(fh.read(max(SIZES) * ROW_BYTES), dtype=np.uint8)

    iron.set_current_device(from_name("npu2", n_cols=None))
    x = np.random.default_rng(0).standard_normal(K).astype(np.float32).astype(bfloat16)
    a_x = iron.tensor(x, dtype=bfloat16, device="npu")

    rows, npu_us, e2e_us = [], [], []
    print(f"{'tile-rows':>10} {'MB':>8} {'npu us':>10} {'e2e us':>10} {'GB/s':>8}")
    for n in SIZES:
        a_w = iron.tensor(blob[: n * ROW_BYTES].copy(), dtype=np.uint8, device="npu")
        c_y = iron.zeros(n * ROWS_PER_TILE, dtype=np.float32, device="npu")
        b = run_iters(lm_head, a_w, a_x, c_y, tile_rows=n, n_cores=CORES,
                      warmup=2, iters=20)
        mb = n * ROW_BYTES / 1e6
        rows.append(n * ROW_BYTES)
        npu_us.append(b.npu.avg_us)
        e2e_us.append(b.e2e.avg_us)
        print(f"{n:10d} {mb:8.2f} {b.npu.avg_us:10.1f} {b.e2e.avg_us:10.1f} "
              f"{mb / b.npu.avg_us * 1e3:8.1f}")

    by = np.array(rows, float)
    for label, t in (("npu", np.array(npu_us)), ("e2e", np.array(e2e_us))):
        slope, fixed = np.polyfit(by, t, 1)
        pred = fixed + slope * by
        r2 = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        print(f"\n{label}:  t = {fixed:.1f} us + bytes / {1 / slope / 1e3:.1f} GB/s"
              f"   (R2 {r2:.4f})")
        work = MEDIAN_LAYER_MB * 1e6 * slope
        print(f"      fixed cost per dispatch      {fixed:8.1f} us")
        print(f"      work in a median layer matmul{work:8.1f} us"
              f"   -> overhead is {100 * fixed / (fixed + work):.0f}% of it")
        for n_disp, what in ((N_FUSED_GROUPS, "fused per-layer groups"),
                             (N_LAYER_MATMULS + 1, "one dispatch per matmul")):
            stream_ms = (328.4 + 270.2) * 1e6 * slope / 1e3
            total = stream_ms + n_disp * fixed / 1e3
            print(f"      {n_disp:3d} dispatches/token ({what:22s}) "
                  f"{total:6.2f} ms -> {1000 / total:5.1f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
