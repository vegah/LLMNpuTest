r"""Diff every q4nx tensor against the upstream bf16 checkpoint.

The chat working is weak evidence: a model with a subtly wrong weight still
produces fluent text. This is the strong version -- every tensor, against
Qwen/Qwen3.5-0.8B, which is the checkpoint FastFlowLM quantised.

A correctly decoded q4 tensor lands at cosine ~0.9975 and a q8 one at ~0.9993;
those are the quantisation floors, not our error. Anything materially below is a
layout bug. For scale, the two layout bugs this found scored 0.138 (q_proj's
query/gate grouping) and 0.030 (the nibble parity).

    ..\NpuEmbeddings\.venv-ref\Scripts\python.exe reference\check_weights.py

Needs the upstream checkpoint, which is not downloaded by default:

    curl -L -o %USERPROFILE%\.cache\openfflm\Qwen3.5-0.8B-upstream\model.safetensors ^
      https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/model.safetensors-00001-of-00001.safetensors
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
from from_q4nx import LINEAR_ATTN, MODEL_DIR, load_config  # noqa: E402
from q4nx import Q4_TILE_BYTES, Q4NX  # noqa: E402

UPSTREAM = MODEL_DIR.parent / "Qwen3.5-0.8B-upstream" / "model.safetensors"
Q4_FLOOR, Q8_FLOOR = 0.995, 0.998


class SafeTensors:
    _DT = {"BF16": bfloat16, "F32": np.float32, "F16": np.float16, "I8": np.int8}

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(n))
            self._start = f.tell()
        self.header.pop("__metadata__", None)

    def get(self, key: str) -> np.ndarray:
        e = self.header[key]
        a, b = e["data_offsets"]
        with self.path.open("rb") as f:
            f.seek(self._start + a)
            raw = f.read(b - a)
        v = np.frombuffer(raw, dtype=self._DT[e["dtype"]]).reshape(e["shape"])
        return v.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> int:
    if not UPSTREAM.exists():
        print(f"missing {UPSTREAM}\nSee the module docstring for the fetch command.")
        return 2

    config = load_config()
    up = SafeTensors(UPSTREAM)
    f = Q4NX(MODEL_DIR / "model.q4nx")
    hf_to_flm = {v: k for k, v in LINEAR_ATTN.items()}  # unused; kept for symmetry
    del hf_to_flm

    worst, n_ok, n_bad = ("", 1.0), 0, 0
    for i, kind in enumerate(config.layer_types):
        flm, hf = f"model.layers.{i}.", f"model.language_model.layers.{i}."
        pairs = [
            (flm + "mlp.gate_proj.weight", hf + "mlp.gate_proj.weight"),
            (flm + "mlp.up_proj.weight", hf + "mlp.up_proj.weight"),
            (flm + "mlp.down_proj.weight", hf + "mlp.down_proj.weight"),
        ]
        if kind == "linear_attention":
            pairs += [
                (flm + "linear_attn.qkv_proj.weight", hf + "linear_attn.in_proj_qkv.weight"),
                (flm + "self_attn.gate_proj.weight", hf + "linear_attn.in_proj_z.weight"),
                (flm + "linear_attn.ssm_out_proj.weight", hf + "linear_attn.out_proj.weight"),
            ]
        else:
            pairs += [
                (flm + "self_attn.q_proj.weight", hf + "self_attn.q_proj.weight"),
                (flm + "self_attn.k_proj.weight", hf + "self_attn.k_proj.weight"),
                (flm + "self_attn.v_proj.weight", hf + "self_attn.v_proj.weight"),
                (flm + "self_attn.o_proj.weight", hf + "self_attn.o_proj.weight"),
            ]

        for a, b in pairs:
            t = up.get(b)
            g = np.asarray(f.tensor(a, tuple(t.shape)), dtype=np.float32)
            if a.endswith("self_attn.q_proj.weight"):
                # FLM stores [all query | all gate]; transformers groups per head.
                h, d = config.num_attention_heads, config.head_dim
                g = g[np.concatenate(
                    [np.r_[j * d:(j + 1) * d, h * d + j * d:h * d + (j + 1) * d]
                     for j in range(h)]
                )]
            c = cosine(g, t)
            floor = Q4_FLOOR if f.header[a]["shape"][2] == Q4_TILE_BYTES else Q8_FLOOR
            ok = c >= floor
            n_ok, n_bad = n_ok + ok, n_bad + (not ok)
            if c < worst[1]:
                worst = (a, c)
            if not ok:
                print(f"  FAIL {a}  cos {c:+.6f}  floor {floor}")

    print(f"\n{n_ok} tensors at or above the quantisation floor, {n_bad} below")
    print(f"worst: {worst[0]}  cos {worst[1]:+.6f}")
    print("PASS" if n_bad == 0 else "FAIL")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
