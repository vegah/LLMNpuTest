# LLMNpuTest — open kernels for FastFlowLM's closed weights

Can you write your own AMD XDNA2 NPU kernels for a model whose kernels are shipped
as opaque binaries? Yes. This repo does it, end to end: it reads FastFlowLM's
undocumented `q4nx` weight format, runs the model's largest projection on the NPU
with a kernel built from source, and chats.

```
npu      lm_head on the array, 248320 x 1024, 270 MB, 8 cores
> Explain how a neural processing unit differs from a GPU.
[128 tokens, 22.8s, 5.62 tok/s; lm_head on npu 128x at 9.7 ms, 5% of wall]
```

Same harness, NPU off: **4.42 tok/s**. NPU on: **5.62 tok/s**. 1.27×, three runs each.

## Where the research lives

The hardware and toolchain knowledge behind this repo is not here. It is in
**[vegah/Npu-Embeddings](https://github.com/vegah/Npu-Embeddings)** — a shipped
XDNA2 product with 253 self-built xclbins, a C++/XRT runtime, and a documented
catalogue of the traps that make AIE programming hard: the silent architecture
fallback, the L1 budget that is 63 KB and not 64, the DMA stream limits, the
rounding mode that defaults to `floor`.

**If you are reading this, or pointing an LLM at it, read that repo first.** This
one is deliberately thin: code, plus this file and `CLAUDE.md`. Findings that need
more than a comment go there, not here.

## Toolchain

Built with **[MLIR-AIE / IRON](https://github.com/Xilinx/mlir-aie)** — AMD's open,
close-to-metal stack for the AIE array. Docs and getting started:
<https://xilinx.github.io/mlir-aie/>. On Windows, follow
`docs/buildHostWin.md` in that repo; it produces the `iron_env.cmd` this project
sources. Everything here compiles through **Peano** (`llvm-aie`); Chess is not
installed and is not needed.

Measured on: AMD Ryzen AI 9 HX 370, NPU Strix, 8 columns, AIE2P. XRT 2.21.0,
driver 32.0.20102.3930, mlir-aie 1.4.2.dev16+g7e00b57, Peano 21.0.0.2026080301.

## What is on the NPU, and how

The array is 8 columns x 4 compute rows of AIE2P cores. Each core has 64 KB of
local memory and its own vector unit; data reaches it by DMA from DDR through a
shim tile, optionally via a memory tile. You write a small C++ kernel against
`aie_api`, describe the data movement in Python with IRON, and the toolchain emits
an `.xclbin` plus an instruction stream that XRT dispatches.

Three designs, in the order they were built:

### A. `designs/rot13` — the readable proof

Host sends ASCII, one core rewrites it, host prints what comes back. ROT13 is
self-inverse, so running it twice must return the input byte for byte — no
tolerance, and a wrong kernel produces visible garbage rather than a small numeric
error. One column, one worker, two ObjectFifos, a 1024-byte int8 tile,
`aie::vector<int8, 64>` (one full 512-bit register) with range masks and two
selects.

### B. `designs/q4nx_unpack` — 4-bit weights dequantised on the core

AIE2P has **no int4 MAC**. int4 is a storage format, so the dequant has to happen
on the core — if the host widened the weights first, the NPU would read bf16 from
DDR and 4-bit would buy nothing. This takes a real 5120-byte tile out of the model
and produces 8192 bf16 values, **bit-exact** against numpy on all of them. The
unpack is a mask, an `aie::to_float` with a non-zero binary-point shift (which
does the divide by 16 for free), a zip and a mul-add.

### C. `designs/lm_head` — the first real kernel

`y[248320] = W[248320,1024] @ x`, W in q4nx's q8 form. **254M of the 752M MACs a
token costs — 34% — in one dispatch.** 8 cores, 270 MB streamed from DDR, 7.2 ms
at 37.7 GB/s, fp32-exact against the host (cosine 1.00000000, max rel err 3e-06).

|  | time | bytes moved | effective |
|---|---|---|---|
| **NPU, q8** | **7.2 ms** | 270 MB | 37.7 GB/s |
| CPU bf16, 12 threads | 13.4 ms | 509 MB | 37.9 GB/s |
| CPU fp32 | 40.4 ms | 1017 MB | 25.2 GB/s |

The CPU reaches the same raw bandwidth; the win is the quantised format halving
the bytes, and leaving all 12 cores free. The kernel is bandwidth-bound, not
compute-bound — one weight byte per MAC, and a core issues far more MACs per cycle
than the shim delivers bytes.

**Why the layout makes it cheap.** A q8 tile is 32 output rows x 256 K, in two
row-blocks of 16, with the *row* index fastest. So `load_v<32>(scales + kb*32)` is
exactly the 32 output rows in order, and two 16-lane code loads 4096 apart
concatenate to those same 32 rows at one k. No gather, no shuffle, one 32-lane MAC
per activation element. FastFlowLM built the layout for this; we get it for free
by reading it correctly.

## The finding that matters

`lm_head` was chosen because it has the best work-per-dispatch ratio in the model.
That is also why it cannot answer the question that decides the project:

```
lm_head        270.2 MB   in    1 matmul   ->  7166 us of work per dispatch
all 24 layers  328.4 MB   in  186 matmuls  ->    47 us of work per dispatch

measured, designs/lm_head/dispatch_probe.py:
  t = 177.9 us + bytes / 39.3 GB/s   (R2 0.9995, sweep over 10 sizes)
  hw_context switch                    563 us
```

**A layer matmul costs 3x more to issue than to run**, and 12x that again if the
design has to switch. Two `hw_context`s coexist happily, but alternating between
them costs 563 µs — so a hybrid that dispatches ~97 times per token would spend
**54 ms/token switching** against 8.7 ms of work. That path is dead, and it is the
same wall TileFuse hit when it concluded decode stays iGPU-dominated on XDNA2.

Which explains FastFlowLM's shape: a monolithic `layer.xclbin` with a
`gen_layer_seq` entry point. They run the *whole layer* on the array — not because
fusing is a nice optimisation, but because returning to the host between matmuls
costs more than the matmuls. Any open replacement has to do the same, or use one
xclbin with many instruction streams in a single context (the pattern
Npu-Embeddings proved with 16 streams and zero design switches per encode).

So the 1.27× here is the honest ceiling for *this* architecture, and the remaining
66% of the arithmetic needs a different one.

## The q4nx format

`tools/q4nx.py` reads it and documents it. `.q4nx` is a plain safetensors file;
weight tensors are pre-tiled `[N/32][K/256][bytes]`, one tile being 32 output rows
x 256 K in two row-blocks of 16.

```
q4  5120 B  [512 B bf16 d][512 B bf16 m][4096 B nibbles]   w = code*d + m
q8  8704 B  [512 B bf16 scale][8192 B int8]                w = code*scale

meta index = kb*32 + rb*16 + r      weight index i = rb*4096 + k*16 + r
q4 byte, nibble = i >> 1, low nibble when i is even
```

Solved against ground truth, not guessed. The tiling came from `ssm_alpha_proj`,
which layer 0 stores **twice in the same file** — quantised and as bf16 — giving
exact ground truth for 67 KB of download; requiring all 8192 codes of all four
tiles to match pinned the mapping. The rest came from a tensor-by-tensor diff
against `Qwen/Qwen3.5-0.8B`, the checkpoint FastFlowLM quantised.

`embed_tokens` stays bf16 (508 MB, 46% of the file) and `lm_head` is int8, not
int4.

Four conversion differences from transformers, **every one of them silent** —
shapes match, `load_state_dict` is happy, the model runs and talks nonsense:

- `q_proj` is twice as wide (`attn_output_gate`) and groups its fused query and
  gate as whole halves, where transformers groups per head. Cosine 0.138 wrong,
  0.9974 right.
- RMSNorm weights have the `(1 + w)` one already folded in (`model.norm` sits at
  mean 4.31, not 0).
- `ssm_a` is `-exp(A_log)` already evaluated, not `A_log`.
- `conv1d` is stored transposed.

## Layout

```
designs/rot13/              rot13.cc  rot13.py
designs/q4nx_unpack/        q4nx_unpack.cc  q4nx_unpack.py  fixtures/
designs/lm_head/            lm_head.h  lm_head_k0..k3.cc  lm_head.py
                            dispatch_probe.py
tools/q4nx.py               container reader + reference dequant
tools/fetch_q4nx.py         range-fetch single tensors from a 1.1 GB model
tools/export_design.py      build an xclbin + instruction stream (IRON env)
reference/from_q4nx.py      q4nx weights -> transformers model
reference/check_weights.py  every tensor vs the upstream bf16 checkpoint
reference/npu.py            dispatch a prebuilt design through XRT
chat.py                     the chat, NPU or --cpu
verify.py                   runs all three designs, prints PASS/FAIL
```

## Running it

**Two environments, and they do not mix.** IRON needs its own; the reference model
needs torch. This is the split Npu-Embeddings uses, and the reason `reference/npu.py`
talks to XRT directly instead of calling IRON — which also takes IRON's ~465 µs of
per-call Python dispatch out of the per-token path.

```
:: build and verify the designs
call c:\dev\mlir-aie\iron_env.cmd
python tools\fetch_q4nx.py
python verify.py
python tools\export_design.py

:: chat (torch environment)
..\NpuEmbeddings\.venv-ref\Scripts\python.exe chat.py "why is the sky blue?"
..\NpuEmbeddings\.venv-ref\Scripts\python.exe chat.py --cpu "why is the sky blue?"
..\NpuEmbeddings\.venv-ref\Scripts\python.exe reference\check_weights.py
```

`verify.py` prints, per design, the xclbin size and the `aie.dma_bd` / `aie.lock`
counts read out of the placed MLIR — so "we built this" is an artifact, not a
claim.

Weights are **not** redistributed. They live in `~/.cache/openfflm/`:

```
curl -L --create-dirs -o %USERPROFILE%\.cache\openfflm\Qwen3.5-0.8B-NPU2\model.q4nx ^
  https://huggingface.co/FastFlowLM/Qwen3.5-0.8B-NPU2/resolve/main/model.q4nx
```

plus `config.json`, `tokenizer.json`, `tokenizer_config.json` and
`chat_template.jinja` from the same repo. `check_weights.py` additionally wants
`Qwen/Qwen3.5-0.8B`'s `model.safetensors`.

## Scope

FastFlowLM's orchestration is MIT and open; everything that touches the array is
not — `gemm.dll`, `dequant.dll`, `lm_head.dll`, `mha.dll`, `q4_npu_eXpress.dll`
and two opaque xclbins, with every public header pimpl'd. Nothing here is vendored
from it. The format was derived from the published model file and the MIT headers.

Not done: the W4A16 GEMV for the layer projections, the Gated DeltaNet recurrence
(18 of the 24 layers), attention on the array, and the single-xclbin multi-stream
dispatch that the measurement above says is required before any of it pays.
