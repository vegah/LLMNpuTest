r"""Does a DDR round trip inside ONE dispatch have a defined order?

WHY THIS MATTERS
----------------
An all-gather is not expressible as join + forward (granite_gather.py: a fifo
cannot be in two ObjectFifoLinkOps). The remaining route to one dispatch per
layer is a round trip through DDR: cores drain their slices to a scratch buffer,
then refill from it so every core sees the whole vector. The bandwidth is
irrelevant -- 16 KB each way against 49 MB of weights per layer, ~0.07%.

The question is **ordering**. If the refill can be issued before the drain has
landed, the cores read stale memory, and that is a silent data race rather than
an error: the design builds, runs, and returns plausible numbers. Nothing should
be built on this until it is established one way or the other.

THE TEST
--------
The scratch buffer is zeroed by the host. Each core writes a tag only it would
write, the round trip happens, and each core sums what came back.

  * sees the tags  -> the drain is ordered before the refill
  * sees zeros     -> it is not, and the round trip needs explicit
                      synchronisation the TaskGroup does not provide

Two TaskGroups are used, with `finish()` between them, which is the only
sequencing primitive the Runtime offers. If that does not order them, nothing in
this API will.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_roundtrip.py
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

SLICE = 512      # bf16 per core, 1024 B -- clear of trap 14's 128 B
CORES = 4
FULL = SLICE * CORES


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_roundtrip(scratch_o: Out, scratch_i: In, result: Out, *,
                      n_cores: CompileTime[int] = CORES):
    full = SLICE * n_cores
    sl_ty = np.ndarray[(SLICE,), np.dtype[bfloat16]]
    full_ty = np.ndarray[(full,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(32,), np.dtype[np.float32]]
    res_ty = np.ndarray[(32 * n_cores,), np.dtype[np.float32]]

    fill = ExternalFunction("probe_fill",
                            source_file=str(HERE / "granite_probe_fill.cc"),
                            arg_types=[sl_ty, np.int32, np.int32],
                            include_dirs=_include_dirs())
    summ = ExternalFunction("probe_sum",
                            source_file=str(HERE / "granite_probe_sum.cc"),
                            arg_types=[full_ty, o_ty, np.int32],
                            include_dirs=_include_dirs())

    of_part = [ObjectFifo(sl_ty, name=f"rp{c}", depth=1) for c in range(n_cores)]
    of_back = ObjectFifo(full_ty, name="rb", depth=1)     # DDR -> every core
    of_res = [ObjectFifo(o_ty, name=f"rr{c}", depth=1) for c in range(n_cores)]

    def core_body(pout, bin_, rout, k_fill, k_sum, tag):
        pe = pout.acquire(1)
        k_fill(pe, SLICE, tag)
        pout.release(1)                 # -> DDR scratch
        be = bin_.acquire(1)            # <- the whole scratch, back again
        re = rout.acquire(1)
        k_sum(be, re, full)
        bin_.release(1)
        rout.release(1)

    workers = [
        Worker(core_body,
               fn_args=[of_part[c].prod(), of_back.cons(), of_res[c].prod(),
                        fill, summ, c + 1],
               stack_size=0xD00)
        for c in range(n_cores)
    ]

    p_taps = TensorTiler2D.simple_tiler((1, full), (1, SLICE))
    r_taps = TensorTiler2D.simple_tiler((1, 32 * n_cores), (1, 32))

    def sequence(a_so, a_si, c_r, pc, bp, rc):
        # Phase 1: every core's slice lands in the scratch buffer.
        tg1 = TaskGroup()
        for c in range(n_cores):
            pc[c].drain(a_so, tap=p_taps[c], wait=True, group=tg1)
        tg1.finish()
        # Phase 2: the whole scratch goes back out to every core. Whether
        # finish() actually orders this after phase 1 is the entire question.
        tg2 = TaskGroup()
        bp.fill(a_si, group=tg2)
        for c in range(n_cores):
            rc[c].drain(c_r, tap=r_taps[c], wait=True, group=tg2)
        tg2.finish()

    rt = Runtime(sequence, [full_ty, full_ty, res_ty,
                            [f.cons() for f in of_part], of_back.prod(),
                            [f.cons() for f in of_res]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main() -> int:
    iron.set_current_device(from_name("npu2", n_cols=None))
    # One buffer, passed as both the drain target and the refill source.
    scratch = iron.zeros(FULL, dtype=bfloat16, device="npu")
    res = iron.zeros(32 * CORES, dtype=np.float32, device="npu")
    granite_roundtrip(scratch, scratch, res, n_cores=CORES)

    got = res.numpy().reshape(CORES, 32)[:, 0]
    expect = float(SLICE * CORES * (CORES + 1) // 2)
    sc = scratch.numpy().astype(np.float64)

    print(f"DDR round trip in one dispatch   {CORES} cores x {SLICE} bf16")
    print(f"  scratch after the run   : first of each slice "
          f"{[float(sc[c * SLICE]) for c in range(CORES)]}")
    print(f"  expected sum per core   : {expect:.1f}")
    print(f"  got                     : {np.round(got, 1)}")

    wrote = all(abs(float(sc[c * SLICE]) - (c + 1)) < 1e-3 for c in range(CORES))
    read_ok = float(np.abs(got - expect).max()) < 1e-3
    agree = float(np.abs(got - got[0]).max()) < 1e-3
    print(f"  phase 1 wrote the scratch : {'yes' if wrote else 'NO'}")
    print(f"  phase 2 read it back      : {'yes' if read_ok else 'NO'}")
    print(f"  all cores agree           : {'yes' if agree else 'NO'}")
    if wrote and not read_ok:
        print("  -> the write landed but the read did not see it: "
              "TaskGroup.finish() does NOT order a refill after a drain")
    ok = wrote and read_ok and agree
    print(f"  [DDR round trip is ordered within one dispatch]  "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
