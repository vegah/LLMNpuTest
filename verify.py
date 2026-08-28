r"""Run both designs end to end and report what was actually built.

    call c:\dev\mlir-aie\iron_env.cmd
    python verify.py

Wall clock is deliberately not reported. A design switch on this hardware costs
~55 us plus ~286 us per column, which swamps anything either kernel does, so a
timing here would be a number about the driver, not about the kernel.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "tools"))
sys.path.insert(0, str(HERE / "designs" / "rot13"))
sys.path.insert(0, str(HERE / "designs" / "q4nx_unpack"))
sys.path.insert(0, str(HERE / "designs" / "lm_head"))

CACHE = Path.home() / ".npu" / "cache"
XRT_SMI = Path("C:/Windows/System32/AMD/xrt-smi.exe")


def check_device() -> bool:
    """The array must be an 8-column Strix. Everything else assumes AIE2P."""
    if not XRT_SMI.exists():
        print(f"device   xrt-smi not found at {XRT_SMI}")
        return False
    out = subprocess.run(
        [str(XRT_SMI), "examine", "-r", "platform"],
        capture_output=True, text=True,
    ).stdout
    name = re.search(r"Name\s*:\s*(.+)", out)
    cols = re.search(r"Total Columns\s*:\s*(\d+)", out)
    name = name.group(1).strip() if name else "?"
    cols = int(cols.group(1)) if cols else 0
    ok = "Strix" in name and cols == 8
    print(f"device   {name}  {cols} columns  {'OK' if ok else 'UNEXPECTED'}")
    return ok


def provenance(marker: str) -> str:
    """What the toolchain actually emitted, read out of the JIT cache.

    The marker is the kernel source name aiecc copies in beside its output, so
    this finds our build rather than any other design in the shared cache.
    """
    hits = sorted(CACHE.glob(f"*/{marker}"), key=lambda p: p.stat().st_mtime)
    if not hits:
        return "        (not found in the JIT cache)"
    d = hits[-1].parent
    xclbin = d / "final.xclbin"
    placed = (d / "input_with_addresses.mlir").read_text(errors="ignore")
    cols = len(set(re.findall(r"aie\.tile\((\d+),\s*[2-5]\)", placed)))
    return (
        f"        {d.name}  xclbin {xclbin.stat().st_size / 1024:.1f} KB  "
        f"insts {(d / 'insts.bin').stat().st_size} B  "
        f"columns {cols}  dma_bd {placed.count('aie.dma_bd')}  "
        f"locks {placed.count('aie.lock')}"
    )


def main() -> int:
    import aie.iron as iron
    from aie.iron.device import from_name

    ok = check_device()

    # Trap 1: without this IRON silently falls back to aie2/NPU1. No error.
    iron.set_current_device(from_name("npu2", n_cols=None))

    import numpy as np

    import rot13 as A
    import q4nx_unpack as B

    print("\nA  rot13 -- text in, text out")
    original = A._to_tile(A.DEFAULT_MSG)
    once = A.run_once(original)
    twice = A.run_once(once)
    show = lambda t: t.tobytes().decode("ascii").rstrip()
    print(f'        in  "{show(original)}"')
    print(f'        npu "{show(once)}"')
    print(f'        out "{show(twice)}"')
    a_ok = np.array_equal(twice, original) and not np.array_equal(once, original)
    print(f"        rot13 x2 == identity   {'PASS' if a_ok else 'FAIL'}")
    print(provenance("rot13_1024.cc"))

    print("\nB  q4nx unpack -- 4-bit weights dequantised on the core")
    tile = B.find_tile()
    got, ref = B.run(tile.read_bytes()[: B.q4nx.Q4_TILE_BYTES])
    bad = int((got != ref).sum())
    g = got.astype(np.float32)
    print(f"        {tile.name}")
    print(f"        {B.q4nx.Q4_TILE_BYTES} B -> {got.size} bf16   "
          f"mean {g.mean():+.6f}  std {g.std():.6f}")
    print(f"        core == numpy reference, {got.size - bad}/{got.size} bf16 "
          f"values   {'PASS' if bad == 0 else 'FAIL'}")
    print(provenance("q4nx_unpack_tile.cc"))

    print("")
    print("C  lm_head -- the model's largest projection, on the array")
    import lm_head as C

    c_ok = True
    try:
        c_ok = C.main(["", "--full", "--cores", "8"]) == 0
    except FileNotFoundError:
        print("        model.q4nx not in ~/.cache/openfflm; skipped")
    print(provenance("lmhead_q8_k0.cc"))

    passed = ok and a_ok and bad == 0 and c_ok
    print(f"\n{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
