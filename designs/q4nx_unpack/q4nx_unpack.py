r"""Dequantise one real q4nx weight tile on the NPU.

Takes a tile straight out of FastFlowLM's `model.q4nx` for Qwen3.5-0.8B-NPU2,
DMAs it to one AIE core, unpacks the 4-bit codes and applies scale and minimum
there, and checks the returned bf16 against the numpy reference in tools/q4nx.py.

    call c:\dev\mlir-aie\iron_env.cmd
    python tools\fetch_q4nx.py
    python designs\q4nx_unpack\q4nx_unpack.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.utils import config

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "tools"))
import q4nx  # noqa: E402

NIB_BYTES = q4nx.N_WEIGHTS // 2  # 4096
META = 2 * q4nx.N_META  # 512 bf16: d then m
OUT = q4nx.N_WEIGHTS  # 8192 bf16

# L1 with depth-2 fifos: 2 * (4096 + 512*2 + 8192*2) = 43,008 B, inside the
# 64,512 B budget (trap 3). Two inputs and one output: inside 2-in/2-out
# (trap 3b), which is why d and m ride in one buffer.

DEFAULT_TENSOR = "model.layers.0.mlp.up_proj.weight"


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    return inc


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def q4nx_unpack(nib: In, meta: In, out: Out):
    nib_ty = np.ndarray[(NIB_BYTES,), np.dtype[np.uint8]]
    meta_ty = np.ndarray[(META,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(OUT,), np.dtype[bfloat16]]

    kernel = ExternalFunction(
        "q4nx_unpack_tile",
        source_file=str(HERE / "q4nx_unpack.cc"),
        arg_types=[nib_ty, meta_ty, out_ty],
        include_dirs=_include_dirs(),
    )

    of_nib = ObjectFifo(nib_ty, name="nib", depth=2)
    of_meta = ObjectFifo(meta_ty, name="meta", depth=2)
    of_out = ObjectFifo(out_ty, name="deq", depth=2)

    def core_body(n, s, o, fn):
        a, b, c = n.acquire(1), s.acquire(1), o.acquire(1)
        fn(a, b, c)
        n.release(1)
        s.release(1)
        o.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_nib.cons(), of_meta.cons(), of_out.prod(), kernel],
        stack_size=0xD00,
    )

    def sequence(a_nib, a_meta, c_out, nib_prod, meta_prod, out_cons):
        nib_prod.fill(a_nib)
        meta_prod.fill(a_meta)
        out_cons.drain(c_out, wait=True)

    rt = Runtime(
        sequence,
        [nib_ty, meta_ty, out_ty, of_nib.prod(), of_meta.prod(), of_out.cons()],
    )
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def run(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Dequantise one tile on the NPU. Returns (npu, reference) as bf16."""
    nib, d, m = q4nx.split_q4_tile(raw)
    meta = q4nx.pack_meta(d, m)

    a_nib = iron.tensor(nib, dtype=np.uint8, device="npu")
    a_meta = iron.tensor(meta, dtype=bfloat16, device="npu")
    c_out = iron.zeros(OUT, dtype=bfloat16, device="npu")
    q4nx_unpack(a_nib, a_meta, c_out)

    return c_out.numpy().copy(), q4nx.dequant_q4(nib, d, m)


def find_tile(name: str = DEFAULT_TENSOR) -> Path:
    hits = sorted((HERE / "fixtures").glob(f"{name}*tile.bin"))
    if not hits:
        raise SystemExit(
            f"no fixture for {name}. Run:  python tools\fetch_q4nx.py {name}"
        )
    return hits[0]


def main(argv: list[str]) -> int:
    path = find_tile(argv[1] if len(argv) > 1 else DEFAULT_TENSOR)
    raw = path.read_bytes()[: q4nx.Q4_TILE_BYTES]

    # Trap 1: without this IRON silently falls back to aie2/NPU1. No error.
    iron.set_current_device(from_name("npu2", n_cols=None))

    got, ref = run(raw)
    g, r = got.astype(np.float32), ref.astype(np.float32)
    bad = int((got != ref).sum())

    print(f"tile   {path.name}")
    print(f"       {q4nx.Q4_TILE_BYTES} B -> {OUT} bf16 weights")
    print(f"npu    mean {g.mean():+.6f}  std {g.std():.6f}  "
          f"min {g.min():+.4f}  max {g.max():+.4f}")
    print(f"ref    mean {r.mean():+.6f}  std {r.std():.6f}  "
          f"min {r.min():+.4f}  max {r.max():+.4f}")
    print(f"       max abs err {np.abs(g - r).max():.3e}   "
          f"differing bf16 values {bad}/{OUT}")
    print(f"       [core == numpy reference]  {'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
