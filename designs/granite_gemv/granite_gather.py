r"""All-gather across cores through the memtile, tested in isolation.

WHY THIS EXISTS SEPARATELY
--------------------------
Fusing two projections into one dispatch needs every core to see the whole
intermediate vector, not just the slice it computed. RoPE was the one exception
(per head, so head-local -- see granite_qrope.py); everything else needs a real
gather: cores -> memtile join -> broadcast back.

That is a new dataflow mechanism AND the fused MLP is a new two-phase core
program. Debugging both at once tells you nothing about which is wrong, which is
exactly how the attention bug cost three build cycles to three wrong guesses.
So the mechanism gets its own test first.

THE TEST
--------
Each core fills its slice with a value only that core writes, the gather runs,
and every core sums the whole gathered vector. The check is strict: every core
must report the SAME total, and that total must equal the sum of all the tags
weighted by the slice length. A gather that dropped a core, broadcast only one
core's slice, or returned stale data fails all three ways.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_gather.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (CompileTime, In, ObjectFifo, Out, Program, Runtime,
                      TaskGroup, Worker)
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from granite_gemv import _include_dirs  # noqa: E402

SLICE = 512            # bf16 per core; 1024 B, clear of the 128 B trap 14
# A MemTile has its own DMA channel budget, not just the shim and the cores: an
# 8-way join is refused with "no MemTile has sufficient DMA capacity for 8
# input/1 output channels". Four is what one memtile serves -- which is exactly
# why the memtile leg in granite_gemv32.py is 4 cores per column. A wider gather
# has to be hierarchical: several memtiles joining 4 each, then combined.
CORES = 4
FULL = SLICE * CORES


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_gather(dummy: In, out: Out, *, n_cores: CompileTime[int] = CORES):
    full = SLICE * n_cores
    sl_ty = np.ndarray[(SLICE,), np.dtype[bfloat16]]
    full_ty = np.ndarray[(full,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(32,), np.dtype[np.float32]]
    d_ty = np.ndarray[(SLICE,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(32 * n_cores,), np.dtype[np.float32]]

    fill = ExternalFunction("probe_fill",
                            source_file=str(HERE / "granite_probe_fill.cc"),
                            arg_types=[sl_ty, np.int32, np.int32],
                            include_dirs=_include_dirs())
    summ = ExternalFunction("probe_sum",
                            source_file=str(HERE / "granite_probe_sum.cc"),
                            arg_types=[full_ty, o_ty, np.int32],
                            include_dirs=_include_dirs())

    # The gather itself: cores join their slices into one memtile object, and
    # that object is forwarded straight back out as a broadcast to every core.
    of_join = ObjectFifo(full_ty, name="gj", depth=2)
    parts = of_join.prod().join(
        [c * SLICE for c in range(n_cores)],
        obj_types=[sl_ty] * n_cores,
        names=[f"gp{c}" for c in range(n_cores)],
    )
    of_bcast = of_join.cons().forward(obj_type=full_ty, name="gb", depth=2)

    of_d = ObjectFifo(d_ty, name="gd", depth=1)
    of_o = [ObjectFifo(o_ty, name=f"go{c}", depth=1) for c in range(n_cores)]

    def core_body(part_out, bcast_in, oout, din, k_fill, k_sum, tag):
        din.acquire(1)
        pe = part_out.acquire(1)
        k_fill(pe, SLICE, tag)
        part_out.release(1)          # -> memtile join
        be = bcast_in.acquire(1)     # <- the whole vector, back from the memtile
        oe = oout.acquire(1)
        k_sum(be, oe, full)
        bcast_in.release(1)
        oout.release(1)
        din.release(1)

    workers = [
        Worker(core_body,
               fn_args=[parts[c].prod(), of_bcast.cons(), of_o[c].prod(),
                        of_d.cons(), fill, summ, c + 1],
               stack_size=0xD00)
        for c in range(n_cores)
    ]

    o_taps = TensorTiler2D.simple_tiler((1, 32 * n_cores), (1, 32))

    def sequence(a_d, c_o, dp, oc):
        tg = TaskGroup()
        dp.fill(a_d, group=tg)
        for c in range(n_cores):
            oc[c].drain(c_o, tap=o_taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [d_ty, out_ty, of_d.prod(), [f.cons() for f in of_o]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main() -> int:
    iron.set_current_device(from_name("npu2", n_cols=None))
    d = iron.tensor(np.zeros(SLICE, np.float32).astype(bfloat16),
                    dtype=bfloat16, device="npu")
    c_o = iron.zeros(32 * CORES, dtype=np.float32, device="npu")
    granite_gather(d, c_o, n_cores=CORES)
    got = c_o.numpy().reshape(CORES, 32)

    # Every core writes (c+1) across its SLICE entries, so the total is
    # SLICE * sum(1..CORES).
    expect = float(SLICE * CORES * (CORES + 1) // 2)
    per_core = got[:, 0]
    same = float(np.abs(per_core - per_core[0]).max())
    err = float(np.abs(per_core - expect).max())

    print(f"all-gather through the memtile   {CORES} cores x {SLICE} bf16 "
          f"-> {SLICE * CORES}")
    print(f"  expected total per core : {expect:.1f}")
    print(f"  got                     : {np.round(per_core, 1)}")
    print(f"  cores agree             : {'yes' if same < 1e-3 else f'NO (spread {same})'}")
    print(f"  matches the tag sum     : {'yes' if err < 1e-3 else f'NO (off by {err})'}")
    ok = same < 1e-3 and err < 1e-3
    print(f"  [gather collects every core and broadcasts it back]  "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
