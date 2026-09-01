r"""One granite transformer layer's matmuls, fused into as few dispatches as possible.

WHY
---
Dispatch is ~178 us fixed (0024), and a layer is seven projections. Measured
per-op that is 7 x 40 + 1 = 281 dispatches per token = ~50 ms of doing nothing,
against ~48 ms of actual weight streaming at 44 GB/s. Half the token budget.

WHAT CAN BE FUSED WITHOUT NEW KERNELS
-------------------------------------
A group of projections can share one dispatch iff they share an **input vector**
and a **K**. The kernel is row-parallel and every tile-row is independent, so
concatenating weights along N is exactly concatenating the outputs -- no new
arithmetic, no new correctness surface.

    q, k, v      share the post-input_layernorm hidden state   -> ONE dispatch
    gate, up     share the post-attention_layernorm hidden     -> ONE dispatch
    o            takes the attention output                    -> alone
    down         takes the SwiGLU output, and K = 8192         -> alone

That is **4 dispatches per layer instead of 7**. Going below 4 requires RMSNorm,
RoPE, attention and SwiGLU on the array, because those sit between the groups --
that is the next piece of work, not this one.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_layer.py            :: fused, layer 0
    python designs\granite_gemv\granite_layer.py --unfused  :: 7 dispatches
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron.device import from_name
from aie.utils.benchmark import run_iters

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "tools"))

import q4nx  # noqa: E402
from granite_gemv import (MODEL, ROWS_PER_TILE, TILE_BYTES, TILE_K,  # noqa: E402
                          projection_shape, reference, tiles_per_call)
from granite_gemv32 import (ROWS_PER_COL, granite_gemv32,  # noqa: E402
                            permute_weights, unpermute_y)

# Which projections share an input vector, and so may share a dispatch.
FUSED_GROUPS = [
    ("qkv", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
    ("o", ["self_attn.o_proj"]),
    ("gate_up", ["mlp.gate_proj", "mlp.up_proj"]),
    ("down", ["mlp.down_proj"]),
]
UNFUSED_GROUPS = [(n.split(".")[-1], [n]) for _, g in FUSED_GROUPS for n in g]


def best_cols(tile_rows: int, max_cols: int = 8) -> int:
    """Widest column count whose core count divides the row count.

    A remainder path is not written, so the row count must divide exactly. This
    is why the group shapes below do not all run at the same width.
    """
    for c in range(max_cols, 0, -1):
        if tile_rows % (c * ROWS_PER_COL) == 0:
            return c
    return 1


def load_group(f, cfg, layer: int, members: list[str]):
    """Concatenate a group's weights along N. Returns (raw, tile_rows, K)."""
    chunks, total_rows, k_all = [], 0, None
    for m in members:
        name = f"model.layers.{layer}.{m}.weight"
        n, k = projection_shape(name, cfg)
        assert k_all in (None, k), f"group mixes K: {k_all} vs {k}"
        k_all = k
        k_tiles = k // TILE_K
        rows = n // ROWS_PER_TILE
        off, _ = f.header[name]["data_offsets"]
        with f.path.open("rb") as fh:
            fh.seek(f._data_start + off)
            chunks.append(fh.read(rows * k_tiles * TILE_BYTES))
        total_rows += rows
    return b"".join(chunks), total_rows, k_all


def run_group(label: str, raw: bytes, tile_rows: int, k: int, x: np.ndarray,
              iters: int, check: bool) -> tuple[float, bool]:
    k_tiles = k // TILE_K
    per_call = tiles_per_call(k_tiles)
    n_entry = k_tiles // per_call
    call_bytes = per_call * TILE_BYTES
    cols = best_cols(tile_rows)
    per_core = tile_rows // (cols * ROWS_PER_COL)

    w = permute_weights(raw, cols, per_core, n_entry, call_bytes)
    a_w = iron.tensor(w, dtype=np.uint8, device="npu")
    a_x = iron.tensor(x, dtype=bfloat16, device="npu")
    c_y = iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu")
    bench = run_iters(granite_gemv32, a_w, a_x, c_y, tile_rows=tile_rows, k=k,
                      n_cols=cols, null=False, warmup=1, iters=iters)
    us = bench.npu.avg_us
    mb = tile_rows * k_tiles * TILE_BYTES / 1e6

    ok = True
    if check:
        got = unpermute_y(c_y.numpy().copy(), cols, per_core)
        ref = reference(raw, x, tile_rows, k_tiles)
        g64, r64 = got.astype(np.float64), ref.astype(np.float64)
        rel = np.abs(g64 - r64).max() / (np.abs(r64).max() + 1e-30)
        cos = float(g64 @ r64 / (np.linalg.norm(g64) * np.linalg.norm(r64) + 1e-30))
        ok = cos > 0.9999999 and rel < 1e-4
        note = f"cos={cos:.8f} rel={rel:.2e}"
    else:
        note = ""
    print(f"  {'ok ' if ok else 'FAIL'} {label:9} {tile_rows * ROWS_PER_TILE:6} rows "
          f"K={k:5} {cols * ROWS_PER_COL:2}c {mb:6.1f}MB {us / 1000:6.2f}ms "
          f"{mb / us * 1e3:5.1f}GB/s {note}")
    return us / 1000.0, ok


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--unfused", action="store_true",
                    help="one dispatch per projection (7) instead of 4")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args(argv[1:])

    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    f = q4nx.Q4NX(MODEL / "model.q4nx")
    n_layers = cfg["num_hidden_layers"]
    iron.set_current_device(from_name("npu2", n_cols=None))

    groups = UNFUSED_GROUPS if a.unfused else FUSED_GROUPS
    mode = "UNFUSED (one dispatch per projection)" if a.unfused else "FUSED by shared input"
    print(f"granite layer {a.layer} matmuls -- {mode}")

    rng = np.random.default_rng(0)
    total_ms, total_mb, all_ok = 0.0, 0.0, True
    for label, members in groups:
        raw, tile_rows, k = load_group(f, cfg, a.layer, members)
        x = rng.standard_normal(k).astype(np.float32).astype(bfloat16)
        ms, ok = run_group(label, raw, tile_rows, k, x, a.iters, not a.no_check)
        total_ms += ms
        total_mb += len(raw) / 1e6
        all_ok &= ok

    # lm_head runs once per token, not once per layer.
    lm_rows = cfg["vocab_size"] // ROWS_PER_TILE
    lm_mb = lm_rows * (cfg["hidden_size"] // TILE_K) * TILE_BYTES / 1e6
    print(f"\n  layer total   {len(groups)} dispatches  {total_mb:.1f} MB  {total_ms:.2f} ms")
    tok_ms = total_ms * n_layers + 3.5
    print(f"  x{n_layers} layers + lm_head(3.5ms) = {tok_ms:.0f} ms/token "
          f"= {1000 / tok_ms:.1f} tok/s")
    print(f"  (matmuls only; RMSNorm/RoPE/attention/SwiGLU still host-side)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
