r"""granite GQA decode attention on the NPU, checked against numpy.

One q head against a KV cache streamed in blocks, with a flash-style online
softmax so the cache is read exactly once. The KV cache is the one tensor that
grows with the conversation -- 8 kv heads x seq x 64 x 2 x 2 B is 2 MB at seq
1024, per layer, per token -- so it can never be resident and a two-pass softmax
would have to stream it twice.

The test runs several sequence lengths deliberately: at seq <= 32 there is a
single block and the online-merge path never executes, so a kernel that got the
rescaling wrong would still pass. Only seq > 32 exercises it.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_attention.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (Buffer, CompileTime, In, ObjectFifo, Out, Program,
                      Runtime, TaskGroup, Worker)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.utils.benchmark import run_iters

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from granite_gemv import MODEL, _include_dirs  # noqa: E402

# One .cc per entry point -- see granite_attention.h.
SRC_BLOCK = str(HERE / "granite_attn_block.cc")
SRC_FINISH = str(HERE / "granite_attn_finish.cc")
SRC_DUMP = str(HERE / "granite_attn_dump.cc")
HD = 64        # granite head_dim
BLK = 32       # KV positions per call, matches GRANITE_ATTN_BLOCK
# q is 64 bf16 = 128 B. Padded to Q_PAD elements purely to test whether a
# small shim transfer is the reason it arrives as zeros; the kernel still
# reads only the first HD.
Q_PAD = 512


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_attention(q: In, kv: In, out: Out, *, seq: CompileTime[int],
                      raw: CompileTime[bool] = False):
    n_blk = (seq + BLK - 1) // BLK
    q_ty = np.ndarray[(Q_PAD,), np.dtype[bfloat16]]
    kv_ty = np.ndarray[(2 * BLK * HD,), np.dtype[bfloat16]]   # K block then V block
    st_ty = np.ndarray[(HD + 2,), np.dtype[np.float32]]
    # In raw mode the design emits the state itself, so the output element and
    # the host buffer both become float32[66].
    # Padded for the same reason q is: a 128-byte shim transfer arrives as
    # zeros. The kernel writes only the first HD; the rest is never read.
    out_ty = (st_ty if raw
              else np.ndarray[(Q_PAD,), np.dtype[bfloat16]])

    blk = ExternalFunction("granite_attn_block", source_file=SRC_BLOCK,
                           arg_types=[q_ty, kv_ty, st_ty, np.int32, np.int32],
                           include_dirs=_include_dirs())
    fin = ExternalFunction(
        "granite_attn_dump" if raw else "granite_attn_finish",
        source_file=SRC_DUMP if raw else SRC_FINISH,
        arg_types=[st_ty, out_ty], include_dirs=_include_dirs())

    # depth 2, not 1. q is acquired once and held for the whole block
    # loop, so at depth 1 the producer has no free buffer to write into
    # and the consumer sees zeros -- which is exactly what it saw.
    of_q = ObjectFifo(q_ty, name="aq", depth=2)
    of_kv = ObjectFifo(kv_ty, name="akv", depth=2)
    of_o = ObjectFifo(out_ty, name="ao", depth=2)

    def body(qi, kvi, oo, state, kb, fn):
        # q is acquired once and held: every block scores against the same query.
        qe = qi.acquire(1)

        # The first block is peeled out of the loop rather than selected inside
        # it. `for i in range_(n): ... 1 if i == 0 else 0` looks right and is
        # not: range_ yields an MLIR value, so `i == 0` is a Python comparison
        # on a Value object and folds to a constant False. `first` would then
        # never be set, the accumulator would never be zeroed, and the kernel
        # would read an uninitialised state buffer -- which is exactly what it
        # did (cosine -0.05 at one block, where the merge path cannot even run).
        ke = kvi.acquire(1)
        kb(qe, ke, state, BLK, 1)
        kvi.release(1)
        # Emitted only when there is something to iterate. A `range_(0)` still
        # builds an scf.for region, and a value acquired outside it and used
        # around it is the current suspect for q arriving as zeros.
        if n_blk > 1:
            for _ in range_(n_blk - 1):
                ke = kvi.acquire(1)
                kb(qe, ke, state, BLK, 0)
                kvi.release(1)
        oe = oo.acquire(1)
        fn(state, oe)
        oo.release(1)
        qi.release(1)

    # Persistent across calls -- this is what makes the softmax online.
    state = Buffer(np.ndarray[(HD + 2,), np.dtype[np.float32]], name="attn_state")
    w = Worker(body, fn_args=[of_q.cons(), of_kv.cons(), of_o.prod(), state,
                              blk, fin], stack_size=0xD00)

    def seq_fn(a_q, a_kv, c_o, qp, kvp, oc):
        tg = TaskGroup()
        qp.fill(a_q, group=tg)
        kvp.fill(a_kv, group=tg)
        oc.drain(c_o, wait=True, group=tg)
        tg.finish()

    all_kv_ty = np.ndarray[(n_blk * 2 * BLK * HD,), np.dtype[bfloat16]]
    rt = Runtime(seq_fn, [q_ty, all_kv_ty, out_ty,
                          of_q.prod(), of_kv.prod(), of_o.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def main() -> int:
    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    assert hd == HD, f"kernel is built for head_dim {HD}, model says {hd}"
    # q4nx-build folds attention_multiplier so that a standard head_dim**-0.5
    # engine is correct, and the emitted config records the post-fold value.
    scale = cfg["attention_multiplier"]
    assert abs(scale - hd ** -0.5) < 1e-9, (
        f"config attention_multiplier {scale} is not head_dim**-0.5; the kernel's "
        f"GRANITE_ATTN_SCALE would be wrong")

    iron.set_current_device(from_name("npu2", n_cols=None))
    rng = np.random.default_rng(0)
    ok = True
    print(f"granite decode attention on NPU   head_dim {hd}  scale {scale}  "
          f"block {BLK}")

    for seq in (32, 64, 128):
        n_blk = (seq + BLK - 1) // BLK
        q_real = rng.standard_normal(HD).astype(np.float32).astype(bfloat16)
        q = np.zeros(Q_PAD, np.float32).astype(bfloat16)
        q[:HD] = q_real
        K = rng.standard_normal((n_blk * BLK, HD)).astype(np.float32).astype(bfloat16)
        V = rng.standard_normal((n_blk * BLK, HD)).astype(np.float32).astype(bfloat16)

        # Interleave per block: the kernel takes one object holding this block's
        # K followed by this block's V.
        kv = np.concatenate([
            np.concatenate([K[i * BLK:(i + 1) * BLK].reshape(-1),
                            V[i * BLK:(i + 1) * BLK].reshape(-1)])
            for i in range(n_blk)])

        c_o = iron.zeros(Q_PAD, dtype=bfloat16, device="npu")
        b = run_iters(granite_attention,
                      iron.tensor(q, dtype=bfloat16, device="npu"),
                      iron.tensor(kv, dtype=bfloat16, device="npu"), c_o,
                      seq=seq, warmup=1, iters=5)
        got = c_o.numpy().astype(np.float64)[:HD]

        qf, Kf, Vf = (q_real.astype(np.float64), K.astype(np.float64),
                      V.astype(np.float64))
        s = (Kf @ qf) * scale
        p = np.exp(s - s.max())
        ref = (p / p.sum()) @ Vf

        rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
        cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
        # The floor here is `aie::exp2`, not the kernel and not bf16 storage.
        # Measured separately (probe in 0145): exp2<bfloat16> on an fp32 vector
        # has up to 5.5e-02 relative error, an order of magnitude coarser than
        # the 4e-03 bf16 rounding would suggest. Every softmax weight goes
        # through it, so ~5e-02 is what a correct kernel achieves on this
        # primitive. The gate is set from that measurement rather than fitted to
        # the result -- and the cosine, which is far more sensitive to a
        # structurally wrong answer than to a uniformly noisy one, is checked
        # separately and tightly.
        good = rel < 5e-2 and cos > 0.999
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  seq {seq:4} ({n_blk} block"
              f"{'s' if n_blk > 1 else ' '})  cosine {cos:.8f}  "
              f"max rel err {rel:.3e}  ({b.npu.avg_us:.1f} us)")
        if n_blk == 1:
            print(f"         single block: the online-merge path is NOT "
                  f"exercised here")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
