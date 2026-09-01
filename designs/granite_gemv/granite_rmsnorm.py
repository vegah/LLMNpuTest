r"""granite RMSNorm on the NPU, checked against the real norm weights.

    y[c] = x[c] * rsqrt(mean(x^2) + eps) * w[c]

The first non-GEMV granite op on the array. It exists because AMD's
`aie_kernels/aie2p/rms_norm.cc` hardcodes `gamma = 1.0f` and never applies the
per-channel weight -- correct magnitude, wrong value, and nothing about the
shapes would tell you (see granite_rmsnorm.h).

Weights come from the installed model, so this is checked against what the model
actually contains rather than against synthetic data: a norm weight vector is
not uniform, and a kernel that ignored it would still look plausible on ones.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_rmsnorm.py
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
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.utils.benchmark import run_iters

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "tools"))

import q4nx  # noqa: E402
from granite_gemv import MODEL, _include_dirs  # noqa: E402


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_rmsnorm(x: In, w: In, y: Out, *, cols: CompileTime[int]):
    vec_ty = np.ndarray[(cols,), np.dtype[bfloat16]]

    kernel = ExternalFunction(
        "granite_rms_norm",
        source_file=str(HERE / "granite_rmsnorm.cc"),
        arg_types=[vec_ty, vec_ty, vec_ty, np.int32],
        include_dirs=_include_dirs(),
    )

    of_x = ObjectFifo(vec_ty, name="nx", depth=2)
    of_w = ObjectFifo(vec_ty, name="nw", depth=2)
    of_y = ObjectFifo(vec_ty, name="ny", depth=2)

    def core_body(xin, win, yout, fn):
        xe, we, ye = xin.acquire(1), win.acquire(1), yout.acquire(1)
        fn(xe, we, ye, cols)
        xin.release(1)
        win.release(1)
        yout.release(1)

    worker = Worker(core_body,
                    fn_args=[of_x.cons(), of_w.cons(), of_y.prod(), kernel],
                    stack_size=0xD00)

    def sequence(a_x, a_w, c_y, x_prod, w_prod, y_cons):
        tg = TaskGroup()
        x_prod.fill(a_x, group=tg)
        w_prod.fill(a_w, group=tg)
        y_cons.drain(c_y, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [vec_ty, vec_ty, vec_ty,
                            of_x.prod(), of_w.prod(), of_y.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="model.layers.0.input_layernorm.weight")
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args(argv[1:])

    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    cols, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
    f = q4nx.Q4NX(MODEL / "model.q4nx")

    h = f.header[a.tensor]
    assert h["dtype"] == "BF16" and h["shape"] == [cols], h
    off, end = h["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + off)
        w = np.frombuffer(fh.read(end - off), dtype=bfloat16).copy()

    rng = np.random.default_rng(0)
    x = rng.standard_normal(cols).astype(np.float32).astype(bfloat16)

    iron.set_current_device(from_name("npu2", n_cols=None))
    a_x = iron.tensor(x, dtype=bfloat16, device="npu")
    a_w = iron.tensor(w, dtype=bfloat16, device="npu")
    c_y = iron.zeros(cols, dtype=bfloat16, device="npu")
    bench = run_iters(granite_rmsnorm, a_x, a_w, c_y, cols=cols,
                      warmup=1, iters=a.iters)
    got = c_y.numpy().astype(np.float64)

    # Reference in float64, for the reason recorded in 0145: a float32 metric
    # over thousands of terms reports its own rounding as the kernel's error.
    xf, wf = x.astype(np.float64), w.astype(np.float64)
    ref = xf * (1.0 / np.sqrt((xf * xf).mean() + eps)) * wf

    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    # The output is bf16 (8 mantissa bits), so ~2^-9 = 2e-3 is the floor any
    # correct implementation sits on; the gate is set by the storage format.
    ok = cos > 0.99999 and rel < 4e-3

    print(f"granite RMSNorm on NPU   {a.tensor}")
    print(f"  cols {cols}   eps {eps}   weight range "
          f"[{float(wf.min()):.4f}, {float(wf.max()):.4f}]")
    print(f"  npu  {got[:4]}")
    print(f"  ref  {ref[:4]}")
    print(f"  cosine {cos:.8f}   max rel err {rel:.3e}   "
          f"({bench.npu.avg_us:.1f} us)")
    print(f"  [core == host RMSNorm]  {'PASS' if ok else 'FAIL'}")

    # A kernel that ignored the weight would still look plausible unless the
    # weight is non-uniform, so state how much it actually varies.
    print(f"  weight is non-uniform (std {float(wf.std()):.4f}), so ignoring it "
          f"would fail this check")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
