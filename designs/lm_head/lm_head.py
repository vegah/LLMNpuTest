r"""lm_head on the NPU: y[N] = W[N, 1024] @ x[1024], W in q4nx q8 form.

One worker per core. Each core owns a contiguous run of tile-rows, so its weight
stream is a flat slice of the tensor and its outputs are a flat slice of the
logits -- concatenating the cores' outputs gives the logits in order, with no
gather on the host.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\lm_head\lm_head.py --tile-rows 16          :: quick check
    python designs\lm_head\lm_head.py --full                  :: all 248320 rows
"""

from __future__ import annotations

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
from aie.helpers.taplib import TensorTiler2D
from aie.utils import config
from aie.utils.benchmark import run_iters

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "tools"))
import q4nx  # noqa: E402

K = 1024
K_TILES = K // q4nx.TILE_K  # 4 tiles of 256 K
TILE_BYTES = q4nx.Q8_TILE_BYTES  # 8704
ROW_BYTES = K_TILES * TILE_BYTES  # 34816 B per tile-row
ROWS_PER_TILE = q4nx.TILE_ROWS  # 32


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    return inc


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lm_head(w: In, x: In, y: Out, *, tile_rows: CompileTime[int],
            n_cores: CompileTime[int] = 1):
    per_core = tile_rows // n_cores
    tile_ty = np.ndarray[(TILE_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(ROWS_PER_TILE,), np.dtype[np.float32]]
    w_ty = np.ndarray[(tile_rows * ROW_BYTES,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(tile_rows * ROWS_PER_TILE,), np.dtype[np.float32]]

    kernels = [
        ExternalFunction(
            f"lmhead_q8_k{i}",
            source_file=str(HERE / f"lm_head_k{i}.cc"),
            arg_types=[tile_ty, x_ty, acc_ty],
            include_dirs=_include_dirs(),
        )
        for i in range(K_TILES)
    ]

    of_w = [ObjectFifo(tile_ty, name=f"w{c}", depth=2) for c in range(n_cores)]
    of_y = [ObjectFifo(acc_ty, name=f"y{c}", depth=2) for c in range(n_cores)]
    # One activation, broadcast: every core multiplies by the same x. Sixteen
    # private copies would also want sixteen shim MM2S channels, and there are
    # only sixteen on the whole device -- the weights need those.
    of_x = ObjectFifo(x_ty, name="x", depth=1)

    def core_body(win, xin, yout, k0, k1, k2, k3):
        # x is acquired once and held: the same activation feeds every tile-row,
        # so re-streaming it per row would add traffic for nothing.
        xe = xin.acquire(1)
        for _ in range_(per_core):
            ye = yout.acquire(1)
            for fn in (k0, k1, k2, k3):
                we = win.acquire(1)
                fn(we, xe, ye)
                win.release(1)
            yout.release(1)
        xin.release(1)

    workers = [
        Worker(
            core_body,
            fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), *kernels],
            stack_size=0xD00,
        )
        for c in range(n_cores)
    ]

    # Each core takes a contiguous slice, so the access patterns are plain
    # tilings of the two runtime buffers. Explicit taps rather than
    # offset/transfer_len: the flat form emits a BD with sizes [0,0,0,0], which
    # the verifier rejects.
    w_taps = TensorTiler2D.simple_tiler(
        (1, tile_rows * ROW_BYTES), (1, per_core * ROW_BYTES))
    y_taps = TensorTiler2D.simple_tiler(
        (1, tile_rows * ROWS_PER_TILE), (1, per_core * ROWS_PER_TILE))

    def sequence(a_w, a_x, c_y, w_prods, x_prod, y_conss):
        # One TaskGroup, waited on once. Issuing each core's drain with wait=True
        # inside the loop serialises the array: core c+1's weights do not start
        # streaming until core c has finished and drained.
        tg = TaskGroup()
        x_prod.fill(a_x, group=tg)
        for c in range(n_cores):
            w_prods[c].fill(a_w, tap=w_taps[c], group=tg)
            y_conss[c].drain(c_y, tap=y_taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
        [w_ty, x_ty, y_ty,
         [f.prod() for f in of_w], of_x.prod(), [f.cons() for f in of_y]],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def reference(raw: bytes, x: np.ndarray, tile_rows: int,
              chunk: int = 256) -> np.ndarray:
    """Same GEMV on the host, from the same bytes, in float32.

    Chunked over tile-rows: the full weight matrix is 248320 x 1024, which is
    1 GB once dequantised to float32 and pointless to hold all at once.
    """
    xf = x.astype(np.float32)
    out = np.empty(tile_rows * ROWS_PER_TILE, np.float32)
    b_all = np.frombuffer(raw, dtype=np.uint8).reshape(tile_rows * K_TILES, TILE_BYTES)
    for lo in range(0, tile_rows, chunk):
        n = min(chunk, tile_rows - lo)
        b = b_all[lo * K_TILES:(lo + n) * K_TILES]
        w = q4nx._untile(q4nx._q8_tiles(b), n, K_TILES)
        out[lo * ROWS_PER_TILE:(lo + n) * ROWS_PER_TILE] = w.astype(np.float32) @ xf
    return out


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-rows", type=int, default=16)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--full", action="store_true", help="all 248320 rows")
    ap.add_argument("--x", default="random", help="random | ones | onehot:<k>")
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args(argv[1:])

    f = q4nx.Q4NX(Path.home() / ".cache" / "openfflm" / "Qwen3.5-0.8B-NPU2" / "model.q4nx")
    n_all = f.header["lm_head.weight"]["shape"][0]
    tile_rows = n_all if a.full else a.tile_rows
    if tile_rows % a.cores:
        raise SystemExit(f"{tile_rows} tile-rows does not divide over {a.cores} cores")

    first, _ = f.header["lm_head.weight"]["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + first)
        raw = fh.read(tile_rows * ROW_BYTES)

    if a.x == "ones":
        x = np.ones(K, np.float32).astype(bfloat16)
    elif a.x.startswith("onehot"):
        x = np.zeros(K, np.float32)
        x[int(a.x.split(":")[1]) if ":" in a.x else 0] = 1.0
        x = x.astype(bfloat16)
    else:
        x = np.random.default_rng(0).standard_normal(K).astype(np.float32).astype(bfloat16)

    # Trap 1: without this IRON silently falls back to aie2/NPU1. No error.
    iron.set_current_device(from_name("npu2", n_cols=None))

    a_w = iron.tensor(np.frombuffer(raw, dtype=np.uint8).copy(), dtype=np.uint8, device="npu")
    a_x = iron.tensor(x, dtype=bfloat16, device="npu")
    c_y = iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu")
    bench = run_iters(lm_head, a_w, a_x, c_y,
                      tile_rows=tile_rows, n_cores=a.cores,
                      warmup=1, iters=a.iters)
    got = c_y.numpy().copy()
    ref = reference(raw, x, tile_rows)
    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))

    print(f"rows      {tile_rows * ROWS_PER_TILE}  ({tile_rows} tile-rows, "
          f"{a.cores} core{'s' if a.cores != 1 else ''}, "
          f"{tile_rows * ROW_BYTES / 1e6:.1f} MB of weights)")
    print(f"npu       {got[:4]}")
    print(f"ref       {ref[:4]}")
    print(f"          cosine {cos:+.8f}   max rel err {rel:.3e}")
    if bench.npu is not None:
        mb = tile_rows * ROW_BYTES / 1e6
        print(f"npu       {bench.npu.avg_us / 1000:.2f} ms   "
              f"{mb / bench.npu.avg_us * 1e3:.1f} GB/s of weights"
              f"   ({2 * tile_rows * ROWS_PER_TILE * K / bench.npu.avg_us / 1e3:.1f} GFLOP/s)")
        print(f"e2e       {bench.e2e.avg_us / 1000:.2f} ms")
    ok = cos > 0.9999999 and rel < 1e-4
    print(f"          [core == host GEMV]  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
