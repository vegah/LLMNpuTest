r"""The whole granite MLP block in ONE dispatch: gate, up, SwiGLU, down.

    h    = silu(gate @ x) * (up @ x)     8192 wide, split 1024 per core
    out  = down @ h                      2560 wide

The first three ops are core-local (SwiGLU pairs gate[i] with up[i] at the same
index). `down_proj` is not: it consumes the whole 8192-wide intermediate, so
every core needs what all eight produced.

THE GATHER GOES THROUGH DDR
---------------------------
`join()` into a memtile and `forward()` back out is not expressible -- a fifo
cannot be in two ObjectFifoLinkOps (granite_gather.py). The route that works is
a round trip through DDR inside the same dispatch, which granite_roundtrip.py
established is ordered by `TaskGroup.finish()` (20/20 runs). It costs 16 KB each
way against 39 MB of MLP weights: ~0.04%.

TWO THINGS THAT WOULD DEADLOCK OR OVERFLOW
------------------------------------------
* **The weight stream is filled twice**, gate|up in phase 1 and down in phase 2.
  One fill spanning both would not complete until the core consumed its phase-2
  objects, which it cannot do until after `tg1.finish()` -- which is waiting for
  that fill. A deadlock, not an error.
* **One fifo carries x then h.** A compute tile has 2 input DMA channels and the
  weights need one; a third for h does not exist. The activation fifo is sized
  for the larger of the two (8192 bf16) and filled twice: x padded in phase 1,
  h in phase 2.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_mlp_full.py
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
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.benchmark import run_iters

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "tools"))
import q4nx  # noqa: E402
from granite_gemv import (MODEL, ROWS_PER_TILE, TILE_BYTES, TILE_K,  # noqa: E402
                          _include_dirs, projection_shape, reference)

PER_CALL = 2
CORES = 8


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_mlp_full(w: In, x: In, scratch_o: Out, scratch_i: In, out: Out, *,
                     hidden: CompileTime[int], inter: CompileTime[int],
                     n_cores: CompileTime[int] = CORES):
    gu_tiles = hidden // TILE_K            # 10 for K = 2560
    dn_tiles = inter // TILE_K             # 32 for K = 8192
    gu_entry = gu_tiles // PER_CALL        # 5
    dn_entry = dn_tiles // PER_CALL        # 16
    call_bytes = PER_CALL * TILE_BYTES

    gu_rows = inter // ROWS_PER_TILE       # 256 tile-rows of gate (and of up)
    dn_rows = hidden // ROWS_PER_TILE      # 80 tile-rows of down
    gu_per = gu_rows // n_cores            # 32
    dn_per = dn_rows // n_cores            # 10
    h_slice = gu_per * ROWS_PER_TILE       # 1024 of the intermediate
    o_slice = dn_per * ROWS_PER_TILE       # 320 of the output

    tile_ty = np.ndarray[(call_bytes,), np.dtype[np.uint8]]
    # One activation fifo, sized for the larger payload and filled twice.
    act_ty = np.ndarray[(inter,), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(h_slice,), np.dtype[np.float32]]
    h_ty = np.ndarray[(h_slice,), np.dtype[bfloat16]]
    # The output element is h_slice wide, not o_slice, so it matches the
    # accumulator type the GEMV entry points declare. An ExternalFunction has one
    # declared signature, and phase 1 writes a 1024-float accumulator while phase
    # 2 writes a 320-float result; declaring a second set of entry points for the
    # narrower type would mean two ExternalFunctions on one .cc, i.e. duplicate
    # symbols. Only the first o_slice values are meaningful; the host slices
    # them out, at a cost of ~22 KB of DMA per dispatch against 39 MB of weights.
    o_ty = np.ndarray[(h_slice,), np.dtype[np.float32]]
    hfull_ty = np.ndarray[(inter,), np.dtype[bfloat16]]
    gu_bytes = 2 * gu_rows * gu_tiles * TILE_BYTES
    dn_bytes = dn_rows * dn_tiles * TILE_BYTES
    w_ty = np.ndarray[(gu_bytes + dn_bytes,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(n_cores * h_slice,), np.dtype[np.float32]]

    gemv = [ExternalFunction(f"granite_mlp_g{i}",
                             source_file=str(HERE / f"granite_mlp_g{i}.cc"),
                             arg_types=[tile_ty, act_ty, acc_ty, np.int32],
                             include_dirs=_include_dirs())
            for i in range(dn_entry)]
    swiglu = ExternalFunction("granite_swiglu_f32",
                              source_file=str(HERE / "granite_swiglu_f32.cc"),
                              arg_types=[acc_ty, acc_ty, h_ty, np.int32],
                              include_dirs=_include_dirs())

    of_w = [ObjectFifo(tile_ty, name=f"fw{c}", depth=2) for c in range(n_cores)]
    of_h = [ObjectFifo(h_ty, name=f"fh{c}", depth=1) for c in range(n_cores)]
    of_o = [ObjectFifo(o_ty, name=f"fo{c}", depth=1) for c in range(n_cores)]
    of_act = ObjectFifo(act_ty, name="fa", depth=1)

    def core_body(win, actin, hout, oout, g_acc, u_acc, sw, *gs):
        # ---- phase 1: gate, up, SwiGLU. All core-local. ----
        xe = actin.acquire(1)
        for acc in (g_acc, u_acc):
            for r in range_(gu_per):
                for fn in gs[:gu_entry]:
                    we = win.acquire(1)
                    fn(we, xe, acc, r)
                    win.release(1)
        he = hout.acquire(1)
        sw(g_acc, u_acc, he, h_slice)
        hout.release(1)              # -> DDR scratch
        actin.release(1)

        # ---- phase 2: down, over the gathered intermediate ----
        be = actin.acquire(1)        # <- the whole 8192, back from DDR
        oe = oout.acquire(1)
        for r in range_(dn_per):
            for fn in gs:
                we = win.acquire(1)
                fn(we, be, oe, r)
                win.release(1)
        oout.release(1)
        actin.release(1)

    workers = []
    for c in range(n_cores):
        g_acc = Buffer(np.ndarray[(h_slice,), np.dtype[np.float32]], name=f"fg{c}")
        u_acc = Buffer(np.ndarray[(h_slice,), np.dtype[np.float32]], name=f"fu{c}")
        workers.append(Worker(
            core_body,
            fn_args=[of_w[c].cons(), of_act.cons(), of_h[c].prod(),
                     of_o[c].prod(), g_acc, u_acc, swiglu, *gemv],
            stack_size=0xD00))

    gu_row_b = gu_tiles * TILE_BYTES
    dn_row_b = dn_tiles * TILE_BYTES
    gu_taps = TensorTiler2D.simple_tiler((1, gu_bytes), (1, 2 * gu_per * gu_row_b))
    # down's weights start AFTER gate|up in the same buffer, so the tap needs
    # that offset. simple_tiler over (1, dn_bytes) would index from 0 and feed
    # phase 2 the gate weights -- which is not an error, just a different and
    # wrong matmul.
    dn_taps = [TensorAccessPattern((1, gu_bytes + dn_bytes),
                                   gu_bytes + c * dn_per * dn_row_b,
                                   [1, dn_per * dn_row_b], [0, 1])
               for c in range(n_cores)]
    h_taps = TensorTiler2D.simple_tiler((1, inter), (1, h_slice))
    o_taps = TensorTiler2D.simple_tiler((1, n_cores * h_slice), (1, h_slice))

    def sequence(a_w, a_x, a_so, a_si, c_o, wp, ap, hc, oc):
        # Phase 1. The weights are filled in two goes, not one: a single fill
        # spanning both phases could not complete until the core consumed its
        # phase-2 objects, which it cannot do until finish() returns -- and
        # finish() is waiting for the fill. Deadlock, not an error.
        tg1 = TaskGroup()
        ap.fill(a_x, group=tg1)
        for c in range(n_cores):
            wp[c].fill(a_w, tap=gu_taps[c], group=tg1)
            hc[c].drain(a_so, tap=h_taps[c], wait=True, group=tg1)
        tg1.finish()
        # Phase 2: the gathered intermediate goes back out to every core.
        tg2 = TaskGroup()
        ap.fill(a_si, group=tg2)
        for c in range(n_cores):
            wp[c].fill(a_w, tap=dn_taps[c], group=tg2)
            oc[c].drain(c_o, tap=o_taps[c], wait=True, group=tg2)
        tg2.finish()

    rt = Runtime(sequence, [w_ty, act_ty, hfull_ty, hfull_ty, out_ty,
                            [f.prod() for f in of_w], of_act.prod(),
                            [f.cons() for f in of_h], [f.cons() for f in of_o]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main() -> int:
    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    hidden, inter = cfg["hidden_size"], cfg["intermediate_size"]
    names = [f"model.layers.0.mlp.{p}.weight"
             for p in ("gate_proj", "up_proj", "down_proj")]
    f = q4nx.Q4NX(MODEL / "model.q4nx")
    raws = []
    for nm in names:
        n, k = projection_shape(nm, cfg)
        off, _ = f.header[nm]["data_offsets"]
        with f.path.open("rb") as fh:
            fh.seek(f._data_start + off)
            raws.append(fh.read((n // ROWS_PER_TILE) * (k // TILE_K) * TILE_BYTES))

    gu_rows, dn_rows = inter // ROWS_PER_TILE, hidden // ROWS_PER_TILE
    gu_per, dn_per = gu_rows // CORES, dn_rows // CORES
    gu_row_b = (hidden // TILE_K) * TILE_BYTES

    # gate|up interleaved per core (BD sizes cap at 1023 per dimension, so a
    # strided tap is not an option), then down's slices in core order.
    g = np.frombuffer(raws[0], np.uint8).reshape(CORES, gu_per * gu_row_b)
    u = np.frombuffer(raws[1], np.uint8).reshape(CORES, gu_per * gu_row_b)
    w_all = np.concatenate([
        np.ascontiguousarray(np.stack([g, u], axis=1)).reshape(-1),
        np.frombuffer(raws[2], np.uint8)])

    rng = np.random.default_rng(0)
    xf = np.zeros(inter, np.float32)
    xf[:hidden] = rng.standard_normal(hidden)     # padded to the fifo's size
    x = xf.astype(bfloat16)

    iron.set_current_device(from_name("npu2", n_cols=None))
    scratch = iron.zeros(inter, dtype=bfloat16, device="npu")
    h_slice, o_slice = inter // CORES, hidden // CORES
    c_o = iron.zeros(CORES * h_slice, dtype=np.float32, device="npu")
    b = run_iters(granite_mlp_full,
                  iron.tensor(w_all, dtype=np.uint8, device="npu"),
                  iron.tensor(x, dtype=bfloat16, device="npu"),
                  scratch, scratch, c_o,
                  hidden=hidden, inter=inter, n_cores=CORES,
                  warmup=1, iters=10)
    # Only the first o_slice of each core's element carries a result.
    got = c_o.numpy().reshape(CORES, h_slice)[:, :o_slice].reshape(-1).astype(np.float64)

    xh = x[:hidden]
    g32 = reference(raws[0], xh, gu_rows, hidden // TILE_K)
    u32 = reference(raws[1], xh, gu_rows, hidden // TILE_K)
    gb = g32.astype(bfloat16).astype(np.float64)
    ub = u32.astype(bfloat16).astype(np.float64)
    h = ((gb / (1.0 + np.exp(-gb))) * ub).astype(np.float32).astype(bfloat16)
    ref = reference(raws[2], h, dn_rows, inter // TILE_K).astype(np.float64)

    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    # aie::tanh sets the floor, as in granite_mlp.py; down_proj adds ~6e-06.
    ok = rel < 8e-2 and cos > 0.999
    print(f"WHOLE MLP BLOCK, one dispatch   hidden {hidden} inter {inter}  "
          f"{CORES} cores")
    print(f"  gate+up+SwiGLU -> DDR gather -> down")
    print(f"  cosine {cos:.8f}   max rel err {rel:.3e}   {b.npu.avg_us:.1f} us")
    # Check the intermediate separately from the result: if the scratch matches
    # h, phase 1 is right and any error is phase 2's. Guessing which half is at
    # fault is what cost three build cycles on attention.
    sc = scratch.numpy().astype(np.float64)
    h64 = h.astype(np.float64)
    h_rel = np.abs(sc - h64).max() / (np.abs(h64).max() + 1e-30)
    print(f"  intermediate vs reference h : max rel err {h_rel:.3e}"
          f"  {'(phase 1 OK)' if h_rel < 5e-2 else '(PHASE 1 WRONG)'}")
    print(f"  [fused == the three-stage host reference]  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
