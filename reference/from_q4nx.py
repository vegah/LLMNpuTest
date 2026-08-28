r"""Load FastFlowLM's q4nx weights into the reference transformers model.

This is the oracle. It answers one question the NPU work cannot answer on its
own -- whether our reading of `q4nx` is actually right -- by putting our decoded
weights into somebody else's implementation of the architecture and seeing if the
model talks. A format error anywhere shows up as garbage text.

It also settles T53: if the low nibble is the wrong parity, adjacent output rows
swap in every projection in the model, and nothing coherent comes out.

Needs torch, so it runs in the reference env, not the IRON one:

    ..\NpuEmbeddings\.venv-ref\Scripts\python.exe chat.py

Name mapping. FLM keeps its own names for the Gated DeltaNet projections, and
two of them are misleading: `self_attn.gate_proj` on a LINEAR attention layer is
the delta-net output gate (`in_proj_z`), not an attention gate. On full-attention
layers there is no separate gate at all -- `attn_output_gate` is on, so `q_proj`
is 2x wide and carries query and gate together.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from q4nx import Q4NX  # noqa: E402

MODEL_DIR = Path.home() / ".cache" / "openfflm" / "Qwen3.5-0.8B-NPU2"

# Keys FLM adds to config.json that transformers does not know about.
FLM_ONLY = {
    "addr_qk", "addr_kv", "flm_version", "vision_model_weight", "vision_config",
    "architectures", "image_token_id", "video_token_id",
}

# HF suffix -> FLM suffix, for the linear-attention block.
LINEAR_ATTN = {
    "linear_attn.in_proj_qkv.weight": "linear_attn.qkv_proj.weight",
    "linear_attn.in_proj_z.weight": "self_attn.gate_proj.weight",
    "linear_attn.in_proj_b.weight": "linear_attn.ssm_beta_proj.bf16.weight",
    "linear_attn.in_proj_a.weight": "linear_attn.ssm_alpha_proj.bf16.weight",
    "linear_attn.out_proj.weight": "linear_attn.ssm_out_proj.weight",
    "linear_attn.conv1d.weight": "linear_attn.ssm_conv1d.weight",
    "linear_attn.A_log": "linear_attn.ssm_a",
    "linear_attn.dt_bias": "linear_attn.ssm_dt.bias",
    "linear_attn.norm.weight": "linear_attn.ssm_norm.weight",
}


def load_config(model_dir: Path = MODEL_DIR):
    from transformers.models.qwen3_5 import Qwen3_5TextConfig

    raw = json.loads((model_dir / "config.json").read_text())
    return Qwen3_5TextConfig(**{k: v for k, v in raw.items() if k not in FLM_ONLY})


# Two conventions differ from transformers, and both are silent if missed --
# a model with these wrong still runs and still produces fluent-looking text.
#
# 1. Qwen3NextRMSNorm computes x_norm * (1 + w), so its weights are stored
#    centred on ZERO. FLM's are centred on one (mean 1.41 / 1.22 / 4.31 for
#    input, post-attention and final norm), i.e. it folded the +1 in for a
#    runtime that just multiplies. Subtract it back out.
#    Not linear_attn.norm: that is Qwen3NextRMSNormGated, plain w * x, and it
#    measures 0.94 -- already in the right form.
RMSNORM_PLUS_ONE = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight", "model.norm.weight",
)

# 2. `attn_output_gate` makes q_proj twice as wide, carrying query and gate
#    together, and the two disagree on how. transformers views it as
#    (..., n_heads, head_dim*2) and chunks the LAST axis, so the rows are
#    grouped per head: [q_h0 | g_h0 | q_h1 | g_h1 | ...]. FLM stores the two
#    halves whole: [all 2048 query rows | all 2048 gate rows]. Un-permuted this
#    scores cos 0.138 against the upstream bf16 checkpoint; re-grouped, 0.9974,
#    which is the int4 quantisation floor every other tensor sits at.
#
# 3. transformers keeps A_log and computes -A_log.exp(). All 288 of FLM's
#    ssm_a values are negative (-10.58 .. -0.0014); A_log would be positive
#    about 94% of the time, so FLM stores -exp(A_log) already evaluated.


def _flm_name(hf_key: str) -> str:
    """HF state_dict key -> the name FLM stores it under."""
    for hf_suffix, flm_suffix in LINEAR_ATTN.items():
        if hf_key.endswith(hf_suffix):
            return hf_key[: -len(hf_suffix)] + flm_suffix
    return hf_key


def build_state_dict(f: Q4NX, model, config) -> dict[str, torch.Tensor]:
    """One tensor per parameter, at the shape the model declares.

    Driving this off `model.state_dict()` rather than off the file means the
    model's own shapes are the check: a wrong un-tiling, a missed transpose or a
    bad name cannot get past `load_state_dict(strict=True)`.
    """
    out: dict[str, torch.Tensor] = {}
    for key, param in model.state_dict().items():
        if key == "lm_head.weight":
            continue  # tied to embed_tokens; FLM stores a q8 copy we don't need
        name = _flm_name(key)
        if name not in f:
            raise KeyError(f"{key} -> {name}: not in {f.path.name}")

        want = tuple(param.shape)
        if key.endswith("conv1d.weight"):
            # FLM stores [kernel, channels]; torch wants [channels, 1, kernel].
            v = np.asarray(f.tensor(name), dtype=np.float32).T[:, None, :]
        else:
            v = np.asarray(
                f.tensor(name, want if len(want) == 2 else None), dtype=np.float32
            )

        if key.endswith("self_attn.q_proj.weight"):
            h, d = config.num_attention_heads, config.head_dim
            v = v[np.concatenate(
                [np.r_[i * d:(i + 1) * d, h * d + i * d:h * d + (i + 1) * d]
                 for i in range(h)]
            )]

        if key.endswith(RMSNORM_PLUS_ONE):
            v = v - 1.0
        elif key.endswith("linear_attn.A_log"):
            v = np.log(-v)

        if v.shape != want:
            raise ValueError(f"{key}: got {v.shape}, model wants {want}")
        out[key] = torch.from_numpy(np.ascontiguousarray(v)).to(param.dtype)
    return out


def build_model(model_dir: Path = MODEL_DIR, verbose: bool = True):
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

    config = load_config(model_dir)
    config._attn_implementation = "eager"
    if verbose:
        n_lin = sum(t == "linear_attention" for t in config.layer_types)
        print(
            f"config   {config.num_hidden_layers} layers "
            f"({n_lin} gated delta-net, {config.num_hidden_layers - n_lin} full "
            f"attention), hidden {config.hidden_size}, vocab {config.vocab_size}"
        )

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    model.to_empty(device="cpu")

    f = Q4NX(model_dir / "model.q4nx")
    if verbose:
        print(f"weights  {f.path.name}  {len(f.names())} tensors")
    sd = build_state_dict(f, model, config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"missing {missing}, unexpected {unexpected}")
    model.tie_weights()
    model.eval()
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"loaded   {n / 1e6:.0f}M parameters, dequantised to "
              f"{next(model.parameters()).dtype}")
    return model


def load_tokenizer(model_dir: Path = MODEL_DIR):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))
