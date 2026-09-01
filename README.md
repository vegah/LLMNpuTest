# LLMNpuTest — open kernels for FastFlowLM's closed weights

FastFlowLM ships fast XDNA2 inference with everything that touches the NPU as
opaque binaries: `gemm.dll`, `lm_head.dll`, `mha.dll`, two xclbins, every public
header pimpl'd. This repo writes those kernels from source instead, against
FastFlowLM's own undocumented `q4nx` weight files.

It does that for two models, for two different reasons:

| | model | why | state |
|---|---|---|---|
| **Part 1** | Qwen3.5-0.8B | FLM already runs it — can we? | `lm_head` on the array, **1.27x**, chats |
| **Part 2** | granite-4.2-3B | **FLM cannot run it at all** | every op on the array, verified; not yet a token loop |

Part 1 is the proof that the format can be read and the array programmed. Part 2
is the point: granite's geometry matches none of FastFlowLM's shipped designs, so
open kernels are the only route, and **getting it into FastFlowLM is the goal**.

## Where the research lives

Three repos, and this is the thinnest of them.

**[vegah/Npu-Embeddings](https://github.com/vegah/Npu-Embeddings)** holds the
hardware and toolchain knowledge — a shipped XDNA2 product with 253 self-built
xclbins, a C++/XRT runtime, and a catalogue of the traps that make AIE
programming hard: the silent architecture fallback, the L1 budget that is 63 KB
and not 64, the DMA stream limits, the rounding mode that defaults to `floor`.
**If you are reading this, or pointing an LLM at it, read that repo first.**
Findings that need more than a comment go there, not here.

**[Atomic-Germ/q4nx-build](https://github.com/Atomic-Germ/q4nx-build)** is the
open converter that produces the `.q4nx` files both this repo and FastFlowLM
read. Granite support for it is
[PR #11](https://github.com/Atomic-Germ/q4nx-build/pull/11).

**Weights come from there; kernels come from here.**

## Toolchain

Built with **[MLIR-AIE / IRON](https://github.com/Xilinx/mlir-aie)** — AMD's open,
close-to-metal stack for the AIE array. Docs:
<https://xilinx.github.io/mlir-aie/>. On Windows follow `docs/buildHostWin.md`
there; it produces the `iron_env.cmd` this project sources. Everything compiles
through **Peano** (`llvm-aie`); Chess is not installed and is not needed.

Measured on: AMD Ryzen AI 9 HX 370, NPU Strix, 8 columns, AIE2P. XRT 2.21.0,
driver 32.0.20102.3930, mlir-aie 1.4.2.dev16+g7e00b57, Peano 21.0.0.2026080301.

The array is **8 columns x 4 compute rows** = 32 AIE2P cores. Each has 64 KB of
local memory and its own vector unit; data reaches it by DMA from DDR through a
shim tile, optionally via a memory tile. You write a small C++ kernel against
`aie_api`, describe the data movement in Python with IRON, and the toolchain
emits an `.xclbin` plus an instruction stream that XRT dispatches.

---

# Part 1 — Qwen3.5-0.8B: can it be done at all?

**[`FastFlowLM/Qwen3.5-0.8B-NPU2`](https://huggingface.co/FastFlowLM/Qwen3.5-0.8B-NPU2)**,
FastFlowLM's XDNA2 build, one 1.1 GB `.q4nx` file. 24 layers, hidden 1024, vocab
248320, and a hybrid token mixer: 18 layers of Gated DeltaNet (linear attention)
and 6 of gated full attention, in the pattern `3 x DeltaNet -> 1 x attention`.
Projections are 4-bit, `lm_head` is 8-bit, the embedding table stays bf16 and is
46% of the file. Ground truth is
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B), the checkpoint
FastFlowLM quantised.

```
npu      lm_head on the array, 248320 x 1024, 270 MB, 8 cores
> Explain how a neural processing unit differs from a GPU.
[128 tokens, 22.8s, 5.62 tok/s; lm_head on npu 128x at 9.7 ms, 5% of wall]
```

Same harness, NPU off: **4.42 tok/s**. NPU on: **5.62 tok/s**. 1.27x, three runs
each.

### A. `designs/rot13` — the readable proof

Host sends ASCII, one core rewrites it, host prints what comes back. ROT13 is
self-inverse, so running it twice must return the input byte for byte — no
tolerance, and a wrong kernel produces visible garbage rather than a small
numeric error. One column, one worker, two ObjectFifos, a 1024-byte int8 tile,
`aie::vector<int8, 64>` (one full 512-bit register) with range masks and two
selects.

### B. `designs/q4nx_unpack` — 4-bit weights dequantised on the core

AIE2P has **no int4 MAC**. int4 is a storage format, so the dequant has to happen
on the core — if the host widened the weights first, the NPU would read bf16 from
DDR and 4-bit would buy nothing. This takes a real 5120-byte tile out of the model
and produces 8192 bf16 values, **bit-exact** against numpy on all of them.

### C. `designs/lm_head` — the first real kernel

`y[248320] = W[248320,1024] @ x`, W in q4nx's q8 form. **254M of the 752M MACs a
token costs — 34% — in one dispatch.** 8 cores, 270 MB from DDR, 7.2 ms at
37.7 GB/s, fp32-exact against the host (cosine 1.00000000, max rel err 3e-06).

|  | time | bytes moved | effective |
|---|---|---|---|
| **NPU, q8** | **7.2 ms** | 270 MB | 37.7 GB/s |
| CPU bf16, 12 threads | 13.4 ms | 509 MB | 37.9 GB/s |
| CPU fp32 | 40.4 ms | 1017 MB | 25.2 GB/s |

The CPU reaches the same raw bandwidth; the win is the quantised format halving
the bytes, and leaving all 12 cores free.

**Why the layout makes it cheap.** A q8 tile is 32 output rows x 256 K, in two
row-blocks of 16, with the *row* index fastest. So `load_v<32>(scales + kb*32)`
is exactly the 32 output rows in order, and two 16-lane code loads 4096 apart
concatenate to those same rows at one k. No gather, no shuffle, one 32-lane MAC
per activation element. FastFlowLM built the layout for this; we get it for free
by reading it correctly.

### The dispatch finding

`lm_head` has the best work-per-dispatch ratio in the model, which is also why it
cannot answer the question that decides the project:

```
lm_head        270.2 MB   in    1 matmul   ->  7166 us of work per dispatch
all 24 layers  328.4 MB   in  186 matmuls  ->    47 us of work per dispatch

measured, designs/lm_head/dispatch_probe.py:
  t = 177.9 us + bytes / 39.3 GB/s   (R2 0.9995, sweep over 10 sizes)
  hw_context switch                    563 us
```

**A layer matmul costs 3x more to issue than to run**, and 12x that again if the
design has to switch. Two `hw_context`s coexist happily, but alternating between
them costs 563 us — so a hybrid dispatching ~97 times per token would spend
**54 ms/token switching** against 8.7 ms of work.

Which explains FastFlowLM's shape: a monolithic `layer.xclbin` with a
`gen_layer_seq` entry point. They run the *whole layer* on the array — not
because fusing is a nice optimisation, but because returning to the host between
matmuls costs more than the matmuls.

**Part 2 acts on that finding**, and tests two of its conclusions. Whole layers
do fuse. But the paragraph this section used to end on — that 1.27x was "the
honest ceiling for this architecture" — was wrong in one specific way: ~50 GB/s
turned out to be a limit *per agent*, not for the machine.

---

# Part 2 — granite-4.2-3B: the model FastFlowLM cannot run

**The goal is to get granite running in FastFlowLM.** For Qwen above, open
kernels were a demonstration — FLM already runs that model and the 1.27x was a
bonus. Here they are the only route.

FastFlowLM's llama engine whitelists `hidden_size` to {2048, 3072, 4096} and
granite is 2560. Padding up to 3072 gets past that gate and then fails, because
granite needs head_dim 64 at hidden 2560 and *every shipped design at hidden
>= 2560 is head_dim 128*. The two sets are disjoint, hidden only pads upward, and
head_dim cannot be padded at all. Verified on hardware: borrowed xclbins hang,
matched donors segfault. So either AMD ships a `(2560, 8192)` design, or the
kernels get written.

The model: **granite-4.2-3B**, hidden 2560, 40 layers, 40 q heads / 8 kv heads,
head_dim 64, intermediate 8192, weights in q4nx's **q4** form.

The weight file is not a problem —
[q4nx-build PR #11](https://github.com/Atomic-Germ/q4nx-build/pull/11) converts
it. The Granite family is Llama's dense tensor set plus four scalar multipliers,
all four of which fold exactly into the quantised weights (scaling a Q4_1 block's
`d` and `m` by `c > 0` leaves every 4-bit code untouched), so the runtime needs
no change. That PR is validated by unpacking and repacking every model FastFlowLM
ships and requiring byte-identical output: **2374 tensors, 13 models, 6 families,
zero mismatches** — ground truth neither the reader nor the writer produced.

So the weights exist and are correct. What was missing was everything that runs
on the array.

## Every op on the array

`designs/granite_gemv`. Each checked against a host reference built from the same
bytes:

| | check |
|---|---|
| all 8 projection shapes (q, k, v, o, gate, up, down, lm_head) | cosine 1.00000000, **exact** under a one-hot activation |
| RMSNorm, with the per-channel weight | 1.9e-03 — the bf16 output floor |
| RoPE, half-split | 2.7e-03 — the bf16 output floor |
| SwiGLU | 8.1e-03 |
| GQA decode attention, online softmax | 0.9993–0.9998 over 1, 2 and 4 KV blocks |

`onehot` being *exact* is the load-bearing check: it isolates a single k, so a
wrong nibble parity or a swapped row-block shows up as a permutation, not noise —
and a permutation cannot be exact.

## The kernel

`granite_gemv.h`, ~1.2 KB of `.text`. W4A16: 4-bit weights, bf16 activation,
fp32 accumulate.

**The Q4_1 minimum is nearly free.** `w = code*d + m` looks like a second
per-element term; it is not, because `m` is constant across a 32-wide K block and
factors out:

```
y[r] = sum_kb { d[kb][r] * (sum_{k in kb} code[k][r]*x[k])
              + m[kb][r] * (sum_{k in kb} x[k]) }
```

The second term needs one **scalar** sum of x per K block — 8 per tile, against
8192 MACs.

**The layout does the rest.** Within a tile the row index is the fastest axis, so
16 consecutive nibbles are 16 different rows at one k, and they all want scales
that are constant for the whole k-block. Load those once per (row-block,
k-block) and the inner loop is a mask, an `aie::to_float` whose binary-point
shift does the divide by 16 for free, an `interleave_zip` to restore row order,
and a MAC. No gather anywhere.

**Two things the hardware forces.** AIE2P has no fp32 vector multiplier —
`aie::mul(vector<float>, vector<float>)` compiles and returns **zero**, silently
— so each fp32 partial is split into two bf16 halves and both are scaled; 8 + 8
mantissa bits land exactly in the fp32 accumulator. And the default rounding mode
is `floor`, a systematic downward bias baked into every weight unless
`conv_even` is set. That split is the whole error budget: the measured ~6e-06 on
random input is 2 x 2^-17, predicted before it was measured.

**Program memory, not speed, shaped the code.** The core has ~16 KB. The body was
first a template on the K-tile index, so every entry point instantiated its own
4736-byte copy and even one overflowed. Making the index a *runtime* argument
with `__attribute__((noinline))` emits it once (2528 B); making it `inline`
rather than `static` gives it vague linkage so per-translation-unit copies merge
into one COMDAT — which is what lets `down_proj`'s K = 8192 have its eight entry
points at all. Vectorising the scalar `sum of x` halved it again to 1248 B, and
was worth 36% of the runtime.

## Four levers, all measured

| | |
|---|---|
| **memtile leg**, 8 -> 24 cores | **2.24x** (lm_head 8.08 -> 3.4–3.6 ms) |
| **layer fusion**, 9 ops in 3 dispatches | **1.40x** |
| **batching** independent tokens (B=2, B=4) | **1.9–2.7x** per token |
| **hybrid NPU + CPU**, one matmul split | **1.37x**, 74.8 GB/s aggregate |

One shim stream per core caps a design at 8 of the array's 32 cores (9 in / 8 out
of 16 channels). The memtile leg — one stream per *column*, `split()` four ways,
`join()` back — reaches 24. Notably **24 beats 32**: adding the last two columns
costs more than the eight extra cores return.

A layer runs as `[q,k,v,RoPE]` + `o_proj` + `[gate,up,SwiGLU,gather,down]` —
**nine ops in three dispatches**, 2.97 ms against ~4.15 ms unfused. The MLP's
gather goes cores -> DDR -> cores *inside* the dispatch, because a fifo cannot be
both joined into and forwarded out of; that round trip is 16 KB each way against
39 MB of weights.

**Fusion and the memtile leg do not compose.** A fused core needs four DMA
streams where a plain GEMV needs three, and the array has no room for the fourth:
on direct shim that is `2n <= 16`, and through memtiles a column would need
10 in / 10 out against the ~6/6 it has. Core count here is
`min(channel budget, divisors of the work)`, and neither is visible in the kernel.

## The ceiling is not what it looked like

A like-for-like CPU baseline (`cpu_baseline.cpp` — same file, same bytes, AVX2,
24 threads) does lm_head in 3.41 ms at 47.0 GB/s, 89% of its own memory
bandwidth. The NPU reaches 3.4–3.6 ms: **parity**, at far lower power.

But ~50 GB/s is what *one agent* pulls, not what the memory system delivers.
Running both at once, the NPU holds 45.4 GB/s while the CPU keeps going —
**~75 GB/s aggregate**, consistent with LPDDR5X-7500's ~120 GB/s. A hybrid that
splits a matmul between them is 1.37x faster than either alone.

## Three traps that fail open

No error, no warning, plausible output:

- **A 128-byte shim transfer silently delivers zeros.** Attention's `q` and its
  result were both 64 bf16; the kernel computed 32 dot products against a zero
  vector. Every score wrong, all in a plausible range, no permutation structure.
  The same design's 8192 B weight stream was fine throughout. It is the *shim
  transfer* size that matters, not the ObjectFifo element: the GEMV designs have
  128-byte `y` elements and are exact, because there the shim moves the whole
  drain at once.
- **`aie::exp2<bfloat16>` has ~5.5e-02 relative error**, an order of magnitude
  coarser than bf16 rounding and undocumented. It is the accuracy floor of any
  softmax built on it.
- **`aie_kernels/aie2p`'s elementwise references are demos.** `rms_norm.cc`
  hardcodes `gamma = 1.0f` and never applies the weight tensor (granite's ranges
  -15 to +23); `rope.cc` uses interleaved pairs where Llama/Granite use
  half-split `rotate_half` — a different rotation with identical magnitudes;
  `swiglu.cc`'s entry point hardcodes a length of 1024 against granite's 8192.

The attention bug took three wrong guesses and three build cycles before it was
found by dumping intermediate state instead of reasoning: the running max, then
the scores, then what the kernel actually received for `q`. Three probes, one
build each, unambiguous at every step.

## What is still between this and FastFlowLM

The arithmetic is done and checked; the integration is not.

1. **A token loop.** The kernels are validated individually and in fused groups
   against host references. Nothing strings them into a forward pass, so granite
   does not chat on the array.
2. **Dispatch from C++.** These designs are driven from IRON's Python, ~465 us
   per call — fine for a correctness harness, fatal in a token loop.
   `reference/npu.py` shows the pattern: talk to XRT directly with a prebuilt
   xclbin and instruction stream.
3. **An engine to host it.** FastFlowLM's `causal_lm` interface is MIT and open,
   so a granite family can be added without touching the closed DLLs — and such
   an engine is not subject to the hidden-size gate, which lives inside
   `llama_npu.dll`.

None is blocked by anything found here; they are work, not obstacles.

---

# Shared

## The q4nx format

`tools/q4nx.py` reads it and documents it. `.q4nx` is a plain safetensors file;
weight tensors are pre-tiled `[N/32][K/256][bytes]`, one tile being 32 output
rows x 256 K in two row-blocks of 16.

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
against the upstream checkpoint — `reference/check_weights.py`, all 150 quantised
tensors at or above the quantisation floor, worst 0.996116.

Four conversion differences from transformers, **every one of them silent** —
shapes match, `load_state_dict` is happy, the model runs and talks nonsense:

- `q_proj` is twice as wide (`attn_output_gate`) and groups its fused query and
  gate as whole halves, where transformers groups per head. Cosine 0.138 wrong,
  0.9974 right.
- RMSNorm weights have the `(1 + w)` one already folded in (`model.norm` sits at
  mean 4.31, not 0).
- `ssm_a` is `-exp(A_log)` already evaluated, not `A_log`.
- `conv1d` is stored transposed.

Note the shape is not recoverable from the file alone: `gate_proj` and
`down_proj` are both stored `[2560, 5120]`, being 256 tile-rows x 10 K-tiles and
80 x 32. They factor differently, and a GEMV against the wrong factoring is not
an error — just a different and wrong matmul. `(N, K)` comes from `config.json`
and the product is checked against the stored shape.

## Layout

```
designs/rot13/              rot13.cc  rot13.py
designs/q4nx_unpack/        q4nx_unpack.cc  q4nx_unpack.py  fixtures/
designs/lm_head/            lm_head.h  lm_head_k0..k3.cc  lm_head.py
                            dispatch_probe.py
designs/granite_gemv/       granite_gemv.h            W4A16 q4nx-q4 GEMV
                            granite_gemv.py           the design + shape table
                            granite_gemv32.py         memtile leg, 24–32 cores
                            granite_rmsnorm.*         RMSNorm with the weight
                            granite_elementwise.*     RoPE, SwiGLU
                            granite_attention.*       GQA decode, online softmax
                            granite_qkv.py            q+k+v+RoPE fused
                            granite_mlp_full.py       gate+up+SwiGLU+down fused
                            granite_gather.py         all-gather probe (negative)
                            granite_roundtrip.py      DDR round-trip ordering
                            cpu_baseline.cpp          like-for-like CPU reference
tools/q4nx.py               container reader + reference dequant
tools/fetch_q4nx.py         range-fetch single tensors from a 1.1 GB model
tools/export_design.py      build an xclbin + instruction stream (IRON env)
reference/from_q4nx.py      q4nx weights -> transformers model
reference/check_weights.py  every tensor vs the upstream bf16 checkpoint
reference/npu.py            dispatch a prebuilt design through XRT
chat.py                     the chat, NPU or --cpu
verify.py                   runs the Qwen designs, prints PASS/FAIL
```

## Running it

**Two environments, and they do not mix.** IRON needs its own; the reference
model needs torch. This is the split Npu-Embeddings uses, and the reason
`reference/npu.py` talks to XRT directly instead of calling IRON — which also
takes IRON's ~465 us of per-call Python dispatch out of the per-token path.

```
:: Qwen designs: build and verify
call c:\dev\mlir-aie\iron_env.cmd
python tools\fetch_q4nx.py
python verify.py
python tools\export_design.py

:: granite kernels: each prints PASS/FAIL against a host reference
python designs\granite_gemv\granite_gemv.py --all --cores 8
python designs\granite_gemv\granite_rmsnorm.py
python designs\granite_gemv\granite_elementwise.py
python designs\granite_gemv\granite_attention.py
python designs\granite_gemv\granite_qkv.py
python designs\granite_gemv\granite_mlp_full.py

:: chat (torch environment) — Qwen only
..\NpuEmbeddings\.venv-ref\Scripts\python.exe chat.py "why is the sky blue?"
..\NpuEmbeddings\.venv-ref\Scripts\python.exe chat.py --cpu "why is the sky blue?"
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
`Qwen/Qwen3.5-0.8B`'s `model.safetensors`. The granite designs read
`Granite-4.2-3B-NPU2/model.q4nx` from the same cache.

## Scope

FastFlowLM's orchestration is MIT and open; everything that touches the array is
not — `gemm.dll`, `dequant.dll`, `lm_head.dll`, `mha.dll`, `q4_npu_eXpress.dll`
and two opaque xclbins, with every public header pimpl'd. Nothing here is
vendored from it. The format was derived from the published model file and the
MIT headers.

**Not done, Qwen3.5-0.8B:** the Gated DeltaNet recurrence (18 of the 24 layers),
and the single-xclbin multi-stream dispatch that the dispatch finding says is
required before the rest pays.

**Not done, granite-4.2-3B:** a token loop, dispatch from C++ rather than IRON's
Python, and any connection to FastFlowLM's own engine. The arithmetic is
complete and verified; the integration is not started.

## History

Roughly the order things happened, because the code does not read like the path
that produced it — and because most of what was learned came from being wrong.

**Qwen, working outward from nothing.** `rot13` to prove a core could be
programmed at all, `q4nx_unpack` to prove 4-bit weights could be widened on the
core rather than by the host, then `lm_head` as the first real matmul. The
dispatch probe came last and reframed everything: a layer matmul costs 3x more to
issue than to run, so FastFlowLM's monolithic `layer.xclbin` is not an
optimisation but a necessity.

**granite, because it cannot run at all.** The whitelist and the disjoint
head_dim/hidden sets were established first, on hardware, by editing configs and
reading which error came back — a rejected value stops at the gate, an accepted
one fails later. Only then was it worth writing kernels. Converter support went
upstream in parallel.

**The corrections, which are the useful part:**

- **Program memory.** Three theories for why the GEMV overflowed the core —
  entry-point count, loop unrolling, runtime `extract` indices — and all three
  were wrong. Compiling the object directly and reading `llvm-objdump -h` settled
  it in seconds against minutes per build. The body was a template instantiated
  per entry point.
- **A spurious FAIL was the metric, not the kernel.** A float32 cosine over 100k
  terms drifts from 1 for bit-identical vectors. Caught by contradiction: `max
  rel err` read exactly 0.0 at the same time. Fixed in float64, against the
  original unrelaxed gate.
- **"Even a perfect NPU design loses to the CPU" — wrong.** It assumed 39.3 GB/s
  was a hard ceiling and that the kernel was near it. A 20-line null-kernel probe
  showed the weight path sustains 46.3 GB/s and the kernel was using 19.9.
- **"~21 tok/s is physics" — wrong.** That is a limit per agent. Running NPU and
  CPU together reaches ~75 GB/s aggregate.
- **Batching was predicted to saturate at 1.36x and did not** (2.3–2.7x). The
  reasoning assumed B tokens means B times the work per byte; it does not,
  because decoding the weights happens once per byte regardless.
- **The attention bug cost three wrong guesses** — the `range_` first-block
  selector, the build cache, and `iron.Buffer` — each plausible, each costing a
  build, each producing no information. It was found by dumping intermediate
  state: the running max, then the scores, then `q` itself, which arrived as
  zeros.
- **A comment asserting a constraint that does not exist** ("the row index must
  be compile-time") propagated from one design into another, where it turned a
  rolled loop into 320 unrolled calls and overflowed the core.

The pattern in all of them: cheap instruments beat good reasoning, and a
measurement that can distinguish two hypotheses is worth more than three that
cannot. The `granite_gather.py` and `granite_roundtrip.py` probes exist for that
reason — they answer one question each, in isolation, before anything is built on
the answer.

Full task logs, including the failures kept in place, are in
[vegah/Npu-Embeddings](https://github.com/vegah/Npu-Embeddings) under `tasks/`.
