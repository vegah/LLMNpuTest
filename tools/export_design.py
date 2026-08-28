r"""Build an IRON design and export its xclbin + instruction stream.

The two environments do not mix: IRON needs `ironenv` and the reference model
needs torch in `..\NpuEmbeddings\.venv-ref`. Rather than merge them, build here
and dispatch there through XRT directly -- which is the architecture a real
runtime wants anyway. NpuEmbeddings ships exactly this shape (C++ and XRT, no
Python in the process), and it takes IRON's ~465 us of per-call Python dispatch
out of the per-token path.

    call c:\dev\mlir-aie\iron_env.cmd
    python tools\export_design.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "designs" / "lm_head"))

CACHE = Path.home() / ".npu" / "cache"
OUT = REPO / "artifacts"


def find_cache(marker_file: str, *markers: str) -> Path:
    """The JIT cache is shared across every design ever built on this machine.

    Match on the kernel source aiecc copies in beside its output AND on strings
    that are unique to this shape, so a design is never confused with another
    build of the same kernel at a different size.
    """
    hits = []
    for src in CACHE.glob(f"*/{marker_file}"):
        mlir = src.parent / "aie.mlir"
        if not mlir.exists():
            continue
        text = mlir.read_text(errors="ignore")
        if all(m in text for m in markers):
            hits.append(src.parent)
    if not hits:
        raise SystemExit(f"no cache dir with {marker_file} and {markers}")
    return max(hits, key=lambda p: p.stat().st_mtime)


def export_lm_head(cores: int = 8) -> None:
    import aie.iron as iron
    from aie.iron.device import from_name

    import q4nx
    from lm_head import K, ROWS_PER_TILE, ROW_BYTES, lm_head

    f = q4nx.Q4NX(Path.home() / ".cache" / "openfflm" / "Qwen3.5-0.8B-NPU2" / "model.q4nx")
    tile_rows = f.header["lm_head.weight"]["shape"][0]
    n_bytes = tile_rows * ROW_BYTES

    iron.set_current_device(from_name("npu2", n_cols=None))
    first, _ = f.header["lm_head.weight"]["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + first)
        raw = np.frombuffer(fh.read(n_bytes), dtype=np.uint8).copy()

    a_w = iron.tensor(raw, dtype=np.uint8, device="npu")
    a_x = iron.zeros(K, dtype=bfloat16, device="npu")
    c_y = iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu")
    lm_head(a_w, a_x, c_y, tile_rows=tile_rows, n_cores=cores)

    src = find_cache("lmhead_q8_k0.cc", f"memref<{n_bytes}xui8>")
    dst = OUT / "lm_head"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("final.xclbin", "insts.bin"):
        shutil.copy(src / name, dst / name)

    placed = (src / "input_with_addresses.mlir").read_text(errors="ignore")
    (dst / "design.json").write_text(json.dumps({
        "name": "lm_head",
        "tile_rows": tile_rows,
        "n_rows": tile_rows * ROWS_PER_TILE,
        "k": K,
        "cores": cores,
        "buffers": [n_bytes, K * 2, tile_rows * ROWS_PER_TILE * 4],
        "source_cache_dir": src.name,
        "dma_bds": placed.count("aie.dma_bd"),
        "locks": placed.count("aie.lock"),
    }, indent=2))
    print(f"lm_head -> {dst.relative_to(REPO)}  from {src.name}  "
          f"xclbin {(dst / 'final.xclbin').stat().st_size / 1024:.1f} KB  "
          f"insts {(dst / 'insts.bin').stat().st_size} B")


if __name__ == "__main__":
    export_lm_head()
