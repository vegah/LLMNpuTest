# LLMNpuTest

Own kernels for XDNA2, against FastFlowLM's `q4nx` weights. See README.md.

Keep this repo lean: two designs, two tools, one verify script. README.md and
this file are the only prose. Anything that needs more room — research notes,
measurements, open questions — goes to `../NpuEmbeddings/research/`, which
already has the register and the discipline for it.

## Hardware and toolchain

AMD Ryzen AI 9 HX 370, NPU Strix, 8 columns, AIE2P. XRT 2.21.0, driver
32.0.20102.3930. mlir-aie 1.4.2.dev16+g7e00b57, Peano 21.0.0.2026080301.
**Peano only** — `xchesscc` needs Vitis and has no Windows build, so anything
requiring Chess (native bfp16 `mm_bfp.cc`, `chess_*` pragmas) is unreachable.

Two environments, the split NpuEmbeddings uses. **Never mix them.**

```
call c:\dev\mlir-aie\iron_env.cmd                        :: NPU designs
python verify.py

..\NpuEmbeddings\.venv-ref\Scripts\python.exe chat.py    :: torch, CPU oracle
```

Model weights live in `~/.cache/openfflm/`, not in the repo.

## Traps

The full catalogue is `../NpuEmbeddings/CLAUDE.md`, earned the hard way and worth
reading before writing a kernel. The ones that bite here:

- **Device pin.** `iron.set_current_device(from_name("npu2", n_cols=None))`
  before any build. Without it IRON falls back to `aie2`/NPU1 silently — no
  error, wrong `mac_dims`, halved shim DMA burst.
- **fp32 accumulate.** bf16 accumulation re-rounds every step.
- **Rounding.** The AIE default is `floor`, not RNE — a systematic downward bias,
  not symmetric noise. `aie::set_rounding` is core-wide state that leaks between
  kernels, so set it where it matters.
- **L1 budget** `2*(m*k*in + k*n*in + m*n*out) < 64512`. 63 KB, not 64: 1 KB is
  program stack.
- **2 in / 2 out DMA streams per core.** This is why `q4nx_unpack` packs `d` and
  `m` into one buffer instead of streaming them separately.
- **No scalar float in a kernel.** It lowers to `__mulsf3`, measured at 1617x.
- **No fp32 vector multiplier on AIE2P.** `aie::mul(vector<float>, vector<float>)`
  compiles and returns **zero**, silently. Keep multiplies in bf16; where fp32
  precision is needed, split the fp32 value into two bf16 halves and multiply
  both — 8 + 8 mantissa bits land exactly in the fp32 accumulator.
- **Peano drops `chess_*` pragmas and `AIE_PREPARE_FOR_PIPELINING` silently.**
  Also: `aie::downshift` on uint8 lowers to a deprecated intrinsic and the build
  promotes it to an error — mask and use `to_float`'s shift argument instead.
- **IRON compiles the kernel source once per `ExternalFunction`.** Four entry
  points in one `.cc` become four objects that each define all four symbols, and
  the link fails on duplicates. One header, one TU per entry point.
- **16 shim MM2S channels on the whole device.** A per-core stream for anything
  shared (the activation vector) runs out at 16 cores; broadcast it from one
  fifo instead. Past that, weights have to go shim -> memtile -> cores.
- **Never write a device tensor through `.numpy()`.** Build the array host-side
  and pass it to `iron.tensor(arr, dtype=..., device="npu")`. Pass `dtype`
  explicitly; without it the buffer is allocated uint32 and the copy fails.
- **`iron.jit`'s cache key does not inspect module globals.** Anything that
  changes the design must be a `CompileTime` parameter.
- **IRON's declared `arg_types` are cosmetic at the kernel boundary.** A shape
  mismatch against the `.cc` compiles clean and hangs with no diagnostic.

## Dispatch is the budget, not arithmetic

Measured here, and it governs every design decision:

```
t = 177.9 us + bytes / 39.3 GB/s      (R2 0.9995, designs/lm_head/dispatch_probe.py)
hw_context switch                     563 us
median layer matmul                    58 us of work
```

Issuing a layer matmul costs 3x what running it costs, and 12x that again if the
design must switch. Do not add a kernel per operation. Either put a whole layer on
the array in one dispatch (what FastFlowLM's `layer.xclbin` / `gen_layer_seq`
does), or put every shape in one xclbin with per-shape instruction streams in a
single hw_context (what Npu-Embeddings' `export_gemm_rtp.py` does -- 16 streams,
zero design switches per encode).

Two `hw_context`s do coexist on one device; that is not the problem. Alternating
between them is.

## Host-side traps

- **XRT teardown order.** Python's collector frees the device before its buffer
  objects and the process dies with an access violation *after* all work has
  completed and printed. Release explicitly: bos, insts_bo, kernel, context,
  device. `NpuDesign.close()`.
- **`nn.Module` has a `cpu()` method.** Naming a submodule `self.cpu` shadows it
  and the failure surfaces somewhere else entirely.
- **pyxrt is not in either virtualenv.** It ships beside the driver;
  `reference/npu.py` adds `C:\Xilinx\XRT\python` to `sys.path` and
  `C:\Windows\System32\AMD` to the DLL directories.

## The oracle

`chat.py` runs the q4nx weights through transformers' own model, and
`reference/check_weights.py` diffs every tensor against the upstream bf16
checkpoint. Kernel work is checked against numpy, and numpy against these.

Do not trust an end-to-end signal to localise a bug. Both nibble parities gave
degenerate chat output while a second layout fault was also present; the
tensor-by-tensor diff found both in one run. An aggregate cannot localise, and
with two faults present it cannot even detect.

## Not vendored

Nothing from `../FastFlowLM/src/lib/**` or `src/xclbins/**`. The format was
derived from the public model file and the MIT-licensed headers.
