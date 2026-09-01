r"""q_proj and RoPE in ONE dispatch, on the NPU.

Fusing two projections generally needs an all-gather: each core holds only its
slice of the intermediate, and the next matmul needs the whole vector. **RoPE is
the exception.** It is applied per head, so a core that owns whole heads can
rotate its own slice with no communication -- which makes this the one fusion in
a granite layer that costs nothing to build.

granite is head_dim 64 = 2 tile-rows per head, so any core owning an even number
of tile-rows owns whole heads: at 8 cores over q_proj's 80 tile-rows, 10 each =
5 heads.

The saving is one dispatch (~178 us) per layer. The point is as much the pattern
as the number: it is the shape a fully fused layer needs, minus the gather.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_qrope.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (Buffer, CompileTime, In, ObjectFifo, Out, Program,
                      Runtime, TaskGroup, Worker)
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D
from aie.utils.benchmark import run_iters

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "tools"))
import q4nx  # noqa: E402
from granite_gemv import (MODEL, ROWS_PER_TILE, TILE_BYTES, TILE_K,  # noqa: E402
                          _include_dirs, projection_shape, reference)

HD = 64
# cos/sin rides at the end of the x buffer rather than in its own ObjectFifo:
# a compute tile has only 2 input DMA channels and the GEMV already uses both.
# It is also 64 bf16 = 128 B, which would have hit trap 14 anyway.
CS_PAD = 64


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_qrope(w: In, x: In, out: Out, *, tile_rows: CompileTime[int],
                  k: CompileTime[int], n_cores: CompileTime[int] = 8):
    k_tiles = k // TILE_K
    per_call, n_entry = 5, k_tiles // 5
    call_bytes = per_call * TILE_BYTES
    per_core = tile_rows // n_cores
    slice_f = per_core * ROWS_PER_TILE          # floats a core accumulates
    assert slice_f % HD == 0, "a core must own whole heads for RoPE to be local"
    n_heads = slice_f // HD

    tile_ty = np.ndarray[(call_bytes,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(k + CS_PAD,), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(slice_f,), np.dtype[np.float32]]
    o_ty = np.ndarray[(slice_f,), np.dtype[bfloat16]]
    w_ty = np.ndarray[(tile_rows * k_tiles * TILE_BYTES,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(tile_rows * ROWS_PER_TILE,), np.dtype[bfloat16]]

    gemv = [ExternalFunction(f"granite_qgemv_g{i}",
                             source_file=str(HERE / f"granite_qgemv_g{i}.cc"),
                             arg_types=[tile_ty, x_ty, acc_ty, np.int32],
                             include_dirs=_include_dirs())
            for i in range(n_entry)]
    rope = ExternalFunction("granite_qrope",
                            source_file=str(HERE / "granite_qrope.cc"),
                            arg_types=[acc_ty, x_ty, o_ty, np.int32],
                            include_dirs=_include_dirs())

    of_w = [ObjectFifo(tile_ty, name=f"qw{c}", depth=2) for c in range(n_cores)]
    # depth 1 for both: each is acquired once and held for the whole call, so a
    # second buffer buys no overlap and L1 has none to spare. With the weights
    # double-buffered at 5 tiles (51200 B), depth 2 here overflowed the tile --
    # 'Basic sequential allocation failed', whose error text prints the map.
    of_o = [ObjectFifo(o_ty, name=f"qo{c}", depth=1) for c in range(n_cores)]
    of_x = ObjectFifo(x_ty, name="qx", depth=1)

    def core_body(win, xin, oout, acc, rope_fn, *gs):
        xe = xin.acquire(1)
        oe = oout.acquire(1)
        # A Python loop, not range_: `row` has to be a compile-time constant for
        # the offset to fold, and per_core is known here.
        for r in range(per_core):
            for fn in gs:
                we = win.acquire(1)
                fn(we, xe, acc, r)
                win.release(1)
        # The rotation runs on this core's own slice, in L1: the matmul's float32
        # output never leaves the tile between the two ops. That is the whole
        # point of the fusion.
        rope_fn(acc, xe, oe, n_heads)
        oout.release(1)
        xin.release(1)

    workers = []
    for c in range(n_cores):
        acc = Buffer(np.ndarray[(slice_f,), np.dtype[np.float32]], name=f"qacc{c}")
        workers.append(Worker(
            core_body,
            fn_args=[of_w[c].cons(), of_x.cons(), of_o[c].prod(),
                     acc, rope, *gemv],
            stack_size=0xD00))

    w_taps = TensorTiler2D.simple_tiler(
        (1, tile_rows * k_tiles * TILE_BYTES), (1, per_core * k_tiles * TILE_BYTES))
    o_taps = TensorTiler2D.simple_tiler(
        (1, tile_rows * ROWS_PER_TILE), (1, slice_f))

    def sequence(a_w, a_x, c_o, wp, xp, oc):
        tg = TaskGroup()
        xp.fill(a_x, group=tg)
        for c in range(n_cores):
            wp[c].fill(a_w, tap=w_taps[c], group=tg)
            oc[c].drain(c_o, tap=o_taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [w_ty, x_ty, out_ty,
                            [f.prod() for f in of_w], of_x.prod(),
                            [f.cons() for f in of_o]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main() -> int:
    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    name = "model.layers.0.self_attn.q_proj.weight"
    n, k = projection_shape(name, cfg)
    k_tiles, cores = k // TILE_K, 8
    tile_rows = n // ROWS_PER_TILE
    theta, pos = float(cfg["rope_theta"]), 17

    f = q4nx.Q4NX(MODEL / "model.q4nx")
    off, _ = f.header[name]["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + off)
        raw = fh.read(tile_rows * k_tiles * TILE_BYTES)

    rng = np.random.default_rng(0)
    half = HD // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / HD))
    ang = pos * inv
    xf = np.zeros(k + CS_PAD, np.float32)
    xf[:k] = rng.standard_normal(k)
    xf[k:k + half], xf[k + half:k + HD] = np.cos(ang), np.sin(ang)
    x = xf.astype(bfloat16)

    iron.set_current_device(from_name("npu2", n_cols=None))
    c_o = iron.zeros(tile_rows * ROWS_PER_TILE, dtype=bfloat16, device="npu")
    b = run_iters(granite_qrope,
                  iron.tensor(np.frombuffer(raw, np.uint8).copy(),
                              dtype=np.uint8, device="npu"),
                  iron.tensor(x, dtype=bfloat16, device="npu"), c_o,
                  tile_rows=tile_rows, k=k, n_cores=cores, warmup=1, iters=10)
    got = c_o.numpy().astype(np.float64)

    # Reference: the same GEMV on the host, then half-split RoPE per head.
    y = reference(raw, x[:k], tile_rows, k_tiles).astype(np.float64).reshape(-1, HD)
    co, si = np.cos(ang), np.sin(ang)
    ref = np.empty_like(y)
    ref[:, :half] = y[:, :half] * co - y[:, half:] * si
    ref[:, half:] = y[:, half:] * co + y[:, :half] * si
    ref = ref.reshape(-1)

    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    cos_ = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    # The output is bf16, so ~4e-03 is its rounding floor; the GEMV contributes
    # ~6e-06 and the rotation is exact in fp32 until the final narrowing.
    ok = rel < 8e-3 and cos_ > 0.9999
    print(f"q_proj + RoPE fused, one dispatch   N={n} K={k}  {cores} cores, "
          f"{tile_rows // cores * ROWS_PER_TILE // HD} heads each")
    print(f"  cosine {cos_:.8f}   max rel err {rel:.3e}   {b.npu.avg_us:.1f} us")
    print(f"  [fused == GEMV then RoPE on the host]  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
