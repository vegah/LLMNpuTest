r"""q, k, v and RoPE in ONE dispatch, on the NPU.

All three projections consume the same input x, so they share a dispatch with no
gather -- the same property that fused gate and up. RoPE then applies to q and k
but not to v, and a core owns whole heads of each, so the epilogue is core-local
too.

At 8 cores: q is 80 tile-rows (10 each = 5 heads), k and v are 16 each (2 each =
1 head). A core accumulates 320 + 64 + 64 = 448 floats and emits the same in
bf16.

Four ops in one dispatch. With the fused MLP (granite_mlp_full.py) a granite
layer is then: [q,k,v,RoPE] -> attention -> o_proj -> [gate,up,SwiGLU,down],
i.e. four dispatches instead of seven, two of them doing four ops each.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_qkv.py
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
from aie.iron.controlflow import range_
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
CS_PAD = 64    # cos/sin rides at the end of x: a tile has only 2 input channels
CORES = 8


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_qkv(w: In, x: In, out: Out, *, k: CompileTime[int],
                q_rows: CompileTime[int], kv_rows: CompileTime[int],
                n_cores: CompileTime[int] = CORES):
    k_tiles = k // TILE_K
    per_call, n_entry = 5, k_tiles // 5
    call_bytes = per_call * TILE_BYTES
    q_per, kv_per = q_rows // n_cores, kv_rows // n_cores
    rows_per = q_per + 2 * kv_per          # q, then k, then v
    slice_f = rows_per * ROWS_PER_TILE
    q_heads = q_per * ROWS_PER_TILE // HD
    k_heads = kv_per * ROWS_PER_TILE // HD
    v_len = kv_per * ROWS_PER_TILE

    tile_ty = np.ndarray[(call_bytes,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(k + CS_PAD,), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(slice_f,), np.dtype[np.float32]]
    o_ty = np.ndarray[(slice_f,), np.dtype[bfloat16]]
    row_b = k_tiles * TILE_BYTES
    w_ty = np.ndarray[((q_rows + 2 * kv_rows) * row_b,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(n_cores * slice_f,), np.dtype[bfloat16]]

    gemv = [ExternalFunction(f"granite_qgemv_g{i}",
                             source_file=str(HERE / f"granite_qgemv_g{i}.cc"),
                             arg_types=[tile_ty, x_ty, acc_ty, np.int32],
                             include_dirs=_include_dirs())
            for i in range(n_entry)]
    rope = ExternalFunction("granite_qkv_rope",
                            source_file=str(HERE / "granite_qkv_rope.cc"),
                            arg_types=[acc_ty, x_ty, o_ty, np.int32, np.int32,
                                       np.int32],
                            include_dirs=_include_dirs())

    of_w = [ObjectFifo(tile_ty, name=f"qw{c}", depth=2) for c in range(n_cores)]
    of_o = [ObjectFifo(o_ty, name=f"qo{c}", depth=1) for c in range(n_cores)]
    of_x = ObjectFifo(x_ty, name="qx", depth=1)

    def core_body(win, xin, oout, acc, rp, *gs):
        xe = xin.acquire(1)
        oe = oout.acquire(1)
        # One rolled loop over all of q, k and v: the row index picks the window
        # in the accumulator, and it is a runtime value (a comment elsewhere
        # claiming it must be compile-time was wrong, and cost a build).
        for r in range_(rows_per):
            for fn in gs:
                we = win.acquire(1)
                fn(we, xe, acc, r)
                win.release(1)
        rp(acc, xe, oe, q_heads, k_heads, v_len)
        oout.release(1)
        xin.release(1)

    workers = []
    for c in range(n_cores):
        acc = Buffer(np.ndarray[(slice_f,), np.dtype[np.float32]], name=f"qa{c}")
        workers.append(Worker(
            core_body,
            fn_args=[of_w[c].cons(), of_x.cons(), of_o[c].prod(), acc, rope,
                     *gemv],
            stack_size=0xD00))

    # The host interleaves q|k|v per core so this is a contiguous slice; a
    # strided tap would be a BD dimension far past the 1023 cap.
    w_taps = TensorTiler2D.simple_tiler(
        (1, (q_rows + 2 * kv_rows) * row_b), (1, rows_per * row_b))
    o_taps = TensorTiler2D.simple_tiler((1, n_cores * slice_f), (1, slice_f))

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
    names = [f"model.layers.0.self_attn.{p}.weight" for p in ("q_proj", "k_proj", "v_proj")]
    nq, k = projection_shape(names[0], cfg)
    nkv, _ = projection_shape(names[1], cfg)
    k_tiles = k // TILE_K
    q_rows, kv_rows = nq // ROWS_PER_TILE, nkv // ROWS_PER_TILE
    theta, pos = float(cfg["rope_theta"]), 17

    f = q4nx.Q4NX(MODEL / "model.q4nx")
    raws = []
    for nm, rows in zip(names, (q_rows, kv_rows, kv_rows)):
        off, _ = f.header[nm]["data_offsets"]
        with f.path.open("rb") as fh:
            fh.seek(f._data_start + off)
            raws.append(fh.read(rows * k_tiles * TILE_BYTES))

    row_b = k_tiles * TILE_BYTES
    q_per, kv_per = q_rows // CORES, kv_rows // CORES
    # per core: its q slice, then its k slice, then its v slice
    parts = []
    for c in range(CORES):
        for raw, per in zip(raws, (q_per, kv_per, kv_per)):
            parts.append(np.frombuffer(raw, np.uint8)[c * per * row_b:(c + 1) * per * row_b])
    w_all = np.concatenate(parts)

    half = HD // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / HD))
    ang = pos * inv
    xf = np.zeros(k + CS_PAD, np.float32)
    rng = np.random.default_rng(0)
    xf[:k] = rng.standard_normal(k)
    xf[k:k + half], xf[k + half:k + HD] = np.cos(ang), np.sin(ang)
    x = xf.astype(bfloat16)

    iron.set_current_device(from_name("npu2", n_cols=None))
    rows_per = q_per + 2 * kv_per
    slice_f = rows_per * ROWS_PER_TILE
    c_o = iron.zeros(CORES * slice_f, dtype=bfloat16, device="npu")
    b = run_iters(granite_qkv, iron.tensor(w_all, dtype=np.uint8, device="npu"),
                  iron.tensor(x, dtype=bfloat16, device="npu"), c_o,
                  k=k, q_rows=q_rows, kv_rows=kv_rows, n_cores=CORES,
                  warmup=1, iters=10)
    got = c_o.numpy().reshape(CORES, slice_f).astype(np.float64)

    xh = x[:k]
    co, si = np.cos(ang), np.sin(ang)

    def rope(y):
        h = y.reshape(-1, HD)
        r = np.empty_like(h)
        r[:, :half] = h[:, :half] * co - h[:, half:] * si
        r[:, half:] = h[:, half:] * co + h[:, :half] * si
        return r.reshape(-1)

    q = reference(raws[0], xh, q_rows, k_tiles).astype(np.float64)
    kk = reference(raws[1], xh, kv_rows, k_tiles).astype(np.float64)
    v = reference(raws[2], xh, kv_rows, k_tiles).astype(np.float64)
    qn, kn = q_per * ROWS_PER_TILE, kv_per * ROWS_PER_TILE
    ref = np.concatenate([
        np.concatenate([rope(q[c * qn:(c + 1) * qn]),
                        rope(kk[c * kn:(c + 1) * kn]),
                        v[c * kn:(c + 1) * kn]])
        for c in range(CORES)]).reshape(CORES, slice_f)

    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    g1, r1 = got.reshape(-1), ref.reshape(-1)
    cos = float(g1 @ r1 / (np.linalg.norm(g1) * np.linalg.norm(r1) + 1e-30))
    ok = rel < 8e-3 and cos > 0.9999
    print(f"q + k + v + RoPE fused, one dispatch   {CORES} cores, "
          f"{q_per * ROWS_PER_TILE // HD} q heads + {kv_per * ROWS_PER_TILE // HD} kv head each")
    print(f"  cosine {cos:.8f}   max rel err {rel:.3e}   {b.npu.avg_us:.1f} us")
    # v must NOT be rotated: check it separately, or a kernel that rotated
    # everything would still pass on the q and k majority.
    v_rel = np.abs(got[:, qn + kn:] - ref[:, qn + kn:]).max() / (
        np.abs(ref[:, qn + kn:]).max() + 1e-30)
    print(f"  v (unrotated) checked separately: {v_rel:.3e}")
    print(f"  [fused == three GEMVs then RoPE on q and k]  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
