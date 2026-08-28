r"""ROT13 on the NPU -- the readable end-to-end proof.

Host sends ASCII, one AIE core rewrites it, host prints what comes back.
Run it twice and you must get the original text: ROT13 is self-inverse.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\rot13\rot13.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.utils import config

HERE = Path(__file__).parent

# Must match the template instantiation in rot13.cc. IRON's declared arg_types
# are cosmetic at the kernel boundary, so a mismatch compiles clean and hangs.
TILE = 1024

DEFAULT_MSG = "HELLO XDNA2, THIS IS OUR OWN KERNEL"


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    return inc


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def rot13(text_in: In, text_out: Out, *, tile: CompileTime[int] = TILE):
    tile_ty = np.ndarray[(tile,), np.dtype[np.int8]]

    kernel = ExternalFunction(
        f"rot13_{tile}",
        source_file=str(HERE / "rot13.cc"),
        arg_types=[tile_ty, tile_ty],
        include_dirs=_include_dirs(),
    )

    of_in = ObjectFifo(tile_ty, name="txt_in", depth=2)
    of_out = ObjectFifo(tile_ty, name="txt_out", depth=2)

    def core_body(rx, tx, fn):
        a = rx.acquire(1)
        b = tx.acquire(1)
        fn(a, b)
        rx.release(1)
        tx.release(1)

    worker = Worker(
        core_body, fn_args=[of_in.cons(), of_out.prod(), kernel], stack_size=0xD00
    )

    def sequence(src, dst, rx_prod, tx_cons):
        rx_prod.fill(src)
        tx_cons.drain(dst, wait=True)

    rt = Runtime(sequence, [tile_ty, tile_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _to_tile(text: str) -> np.ndarray:
    raw = text.encode("ascii")
    if len(raw) > TILE:
        raise ValueError(f"message is {len(raw)} bytes, tile is {TILE}")
    return np.frombuffer(raw.ljust(TILE, b" "), dtype=np.int8).copy()


def run_once(tile: np.ndarray) -> np.ndarray:
    """One dispatch. Builds host-side, never writes through .numpy() (trap 6b)."""
    src = iron.tensor(tile, dtype=np.int8, device="npu")
    dst = iron.zeros(TILE, dtype=np.int8, device="npu")
    rot13(src, dst, tile=TILE)
    return dst.numpy().copy()


def main(argv: list[str]) -> int:
    msg = " ".join(argv[1:]) or DEFAULT_MSG

    # Trap 1: without this IRON silently falls back to aie2/NPU1. No error.
    iron.set_current_device(from_name("npu2", n_cols=None))

    original = _to_tile(msg)
    once = run_once(original)
    twice = run_once(once)

    show = lambda t: t.tobytes().decode("ascii").rstrip()
    print(f'in : "{show(original)}"')
    print(f'npu: "{show(once)}"')
    print(f'out: "{show(twice)}"')

    ok = np.array_equal(twice, original) and not np.array_equal(once, original)
    print(f"     [rot13 x2 == identity]  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
