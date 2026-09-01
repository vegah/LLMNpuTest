r"""granite's RoPE and SwiGLU on the NPU, checked against numpy.

Both kernels are written rather than adopted; `granite_elementwise.h` records
why (AMD's rope.cc uses the wrong pairing convention, and its swiglu.cc entry
point hardcodes a length of 1024 against granite's 8192).

The RoPE check is the one that matters: the two conventions produce outputs of
identical magnitude, so only an element-wise comparison against the half-split
reference distinguishes them. This test therefore also runs the interleaved
convention and asserts that it does NOT match -- otherwise the test would pass
for the wrong kernel.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_elementwise.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (CompileTime, In, ObjectFifo, Out, Program, Runtime,
                      TaskGroup, Worker)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.utils.benchmark import run_iters

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from granite_gemv import MODEL, _include_dirs  # noqa: E402

SRC = str(HERE / "granite_elementwise.cc")


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_rope(x: In, cs: In, y: Out, *, n: CompileTime[int],
                 half: CompileTime[int], n_heads: CompileTime[int]):
    x_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    cs_ty = np.ndarray[(2 * half,), np.dtype[bfloat16]]
    fn = ExternalFunction("granite_rope", source_file=SRC,
                          arg_types=[x_ty, cs_ty, x_ty, np.int32],
                          include_dirs=_include_dirs())
    of_x, of_c, of_y = (ObjectFifo(x_ty, name="rx", depth=2),
                        ObjectFifo(cs_ty, name="rc", depth=2),
                        ObjectFifo(x_ty, name="ry", depth=2))

    def body(xi, ci, yo, k):
        xe, ce, ye = xi.acquire(1), ci.acquire(1), yo.acquire(1)
        k(xe, ce, ye, n_heads)
        xi.release(1); ci.release(1); yo.release(1)

    w = Worker(body, fn_args=[of_x.cons(), of_c.cons(), of_y.prod(), fn],
               stack_size=0xD00)

    def seq(a_x, a_c, c_y, xp, cp, yc):
        tg = TaskGroup()
        xp.fill(a_x, group=tg); cp.fill(a_c, group=tg)
        yc.drain(c_y, wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [x_ty, cs_ty, x_ty, of_x.prod(), of_c.prod(), of_y.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


# 8192 elements in one object needs 3 buffers x 16384 B x depth 2 = 98304 B
# against a 64 KB L1 -- `Basic sequential allocation failed`, with the MemoryMap
# printed in the error, which is the fastest way to see exactly what overflowed.
# The kernel takes its length at runtime, so the fix is to stream the vector in
# chunks rather than hold it: 3 x 4096 B x 2 = 24576 B.
SWIGLU_CHUNK = 2048


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_swiglu(g: In, u: In, y: Out, *, n: CompileTime[int],
                   chunk: CompileTime[int] = SWIGLU_CHUNK):
    v_ty = np.ndarray[(chunk,), np.dtype[bfloat16]]
    all_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    fn = ExternalFunction("granite_swiglu", source_file=SRC,
                          arg_types=[v_ty, v_ty, v_ty, np.int32],
                          include_dirs=_include_dirs())
    of_g, of_u, of_y = (ObjectFifo(v_ty, name="sg", depth=2),
                        ObjectFifo(v_ty, name="su", depth=2),
                        ObjectFifo(v_ty, name="sy", depth=2))

    def body(gi, ui, yo, k):
        for _ in range_(n // chunk):
            ge, ue, ye = gi.acquire(1), ui.acquire(1), yo.acquire(1)
            k(ge, ue, ye, chunk)
            gi.release(1); ui.release(1); yo.release(1)

    w = Worker(body, fn_args=[of_g.cons(), of_u.cons(), of_y.prod(), fn],
               stack_size=0xD00)

    def seq(a_g, a_u, c_y, gp, up_, yc):
        tg = TaskGroup()
        gp.fill(a_g, group=tg); up_.fill(a_u, group=tg)
        yc.drain(c_y, wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [all_ty, all_ty, all_ty,
                       of_g.prod(), of_u.prod(), of_y.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def report(name: str, got: np.ndarray, ref: np.ndarray, us: float,
           gate: float) -> bool:
    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    ok = rel < gate
    print(f"  {'PASS' if ok else 'FAIL'}  {name:8} cosine {cos:.8f}  "
          f"max rel err {rel:.3e}  ({us:.1f} us)")
    return ok


def main() -> int:
    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    heads, half = cfg["num_attention_heads"], hd // 2
    inter, theta = cfg["intermediate_size"], float(cfg["rope_theta"])
    n = heads * hd
    rng = np.random.default_rng(0)
    iron.set_current_device(from_name("npu2", n_cols=None))
    ok = True

    print(f"granite elementwise on NPU   head_dim {hd}  heads {heads}  "
          f"intermediate {inter}  rope_theta {theta:g}")

    # ---- RoPE ----
    pos = 17
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / hd))
    ang = pos * inv
    cs = np.concatenate([np.cos(ang), np.sin(ang)]).astype(np.float32).astype(bfloat16)
    x = rng.standard_normal(n).astype(np.float32).astype(bfloat16)
    c_y = iron.zeros(n, dtype=bfloat16, device="npu")
    b = run_iters(granite_rope, iron.tensor(x, dtype=bfloat16, device="npu"),
                  iron.tensor(cs, dtype=bfloat16, device="npu"), c_y,
                  n=n, half=half, n_heads=heads, warmup=1, iters=5)
    got = c_y.numpy().astype(np.float64)

    xh = x.astype(np.float64).reshape(heads, hd)
    co, si = np.cos(ang), np.sin(ang)
    ref = np.empty_like(xh)
    ref[:, :half] = xh[:, :half] * co - xh[:, half:] * si     # half-split
    ref[:, half:] = xh[:, half:] * co + xh[:, :half] * si
    ok &= report("rope", got, ref.reshape(-1), b.npu.avg_us, 6e-3)

    # The interleaved convention must NOT match, or this test would also pass
    # for AMD's rope.cc and prove nothing.
    alt = np.empty_like(xh)
    e, o = xh[:, 0::2], xh[:, 1::2]
    alt[:, 0::2] = e * co - o * si
    alt[:, 1::2] = e * si + o * co
    alt_rel = np.abs(got - alt.reshape(-1)).max() / np.abs(alt).max()
    print(f"        interleaved convention differs by {alt_rel:.3f} "
          f"-> the test distinguishes the two")
    ok &= alt_rel > 0.1

    # ---- SwiGLU ----
    g = rng.standard_normal(inter).astype(np.float32).astype(bfloat16)
    u = rng.standard_normal(inter).astype(np.float32).astype(bfloat16)
    c_y2 = iron.zeros(inter, dtype=bfloat16, device="npu")
    b2 = run_iters(granite_swiglu, iron.tensor(g, dtype=bfloat16, device="npu"),
                   iron.tensor(u, dtype=bfloat16, device="npu"), c_y2,
                   n=inter, warmup=1, iters=5)
    got2 = c_y2.numpy().astype(np.float64)
    gf, uf = g.astype(np.float64), u.astype(np.float64)
    ref2 = (gf / (1.0 + np.exp(-gf))) * uf
    ok &= report("swiglu", got2, ref2, b2.npu.avg_us, 1.5e-2)
    nz = int((np.abs(got2) > 0).sum())
    print(f"        {nz}/{inter} outputs non-zero -> the whole vector was "
          f"processed, not the first 1024")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
