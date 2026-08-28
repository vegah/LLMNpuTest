r"""Dispatch a prebuilt design on the NPU through XRT, with no IRON in the loop.

Two reasons this talks to XRT directly rather than calling the IRON design:

  * the environments do not mix -- IRON lives in `ironenv`, the reference model
    needs torch in `..\NpuEmbeddings\.venv-ref`;
  * IRON's Python dispatch costs about 465 us per call on top of the hardware's
    own 178 us (designs/lm_head/dispatch_probe.py). On a token that issues one
    dispatch it is noise; on one that issues a hundred it is the whole budget.

Build the artifacts first, in the IRON environment:

    call c:\dev\mlir-aie\iron_env.cmd
    python tools\export_design.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

ARTIFACTS = Path(__file__).parent.parent / "artifacts"

# pyxrt ships beside the driver, not in either virtualenv.
_XRT_PY = Path("C:/Xilinx/XRT/python")
if _XRT_PY.exists() and str(_XRT_PY) not in sys.path:
    sys.path.insert(0, str(_XRT_PY))
for _d in (r"C:\Windows\System32\AMD", r"C:\Xilinx\XRT"):
    if os.path.isdir(_d):
        os.add_dll_directory(_d)


class NpuDesign:
    """One exported design, resident on the device.

    Weights are uploaded once and stay in a device buffer. They still stream
    from DDR on every token -- 270 MB does not fit in 8 MB of on-chip memory --
    but the host is out of that path, and the hw_context stays loaded so
    repeated calls do not pay a design switch.
    """

    OPCODE = 3  # transaction opcode; args are (opcode, insts, n_insts, *buffers)

    def __init__(self, name: str, device=None):
        import pyxrt

        self.pyxrt = pyxrt
        d = ARTIFACTS / name
        self.meta = json.loads((d / "design.json").read_text())
        self.device = device or pyxrt.device(0)

        xclbin = pyxrt.xclbin(str(d / "final.xclbin"))
        self.device.register_xclbin(xclbin)
        self.context = pyxrt.hw_context(self.device, xclbin.get_uuid())
        kname = xclbin.get_kernels()[0].get_name()
        self.kernel = pyxrt.kernel(self.context, kname)

        insts = np.fromfile(d / "insts.bin", dtype=np.uint8)
        self.insts_bo = pyxrt.bo(self.device, insts.nbytes,
                                 pyxrt.bo.cacheable, self.kernel.group_id(1))
        self.insts_bo.write(insts.tobytes(), 0)
        self.insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.n_insts = insts.nbytes

        # Buffer args start at index 3, after opcode / insts / insts_len.
        self.bos = [
            pyxrt.bo(self.device, nbytes, pyxrt.bo.host_only,
                     self.kernel.group_id(3 + i))
            for i, nbytes in enumerate(self.meta["buffers"])
        ]
        self.calls = 0
        self.seconds = 0.0

    def upload(self, i: int, data: np.ndarray) -> None:
        self.bos[i].write(np.ascontiguousarray(data).tobytes(), 0)
        self.bos[i].sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def run(self) -> None:
        h = self.kernel(self.OPCODE, self.insts_bo, self.n_insts, *self.bos)
        r = h.wait()
        if r != self.pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"NPU returned {r}")

    def close(self) -> None:
        """Release in dependency order.

        Python's collector does not know that a BO outlives nothing and the
        device outlives everything; left to itself it frees the device first and
        the process dies with an access violation during teardown, after all the
        work has completed and printed.
        """
        for attr in ("bos", "insts_bo", "kernel", "context", "device"):
            if hasattr(self, attr):
                delattr(self, attr)

    def download(self, i: int, dtype, count: int) -> np.ndarray:
        self.bos[i].sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return np.frombuffer(self.bos[i].read(count * np.dtype(dtype).itemsize, 0),
                             dtype=dtype)


class NpuLmHead(torch.nn.Module):
    """The tied lm_head, computed on the array. Decode only.

    Prefill wants logits at many positions at once, which is a GEMM and not what
    this kernel is, so it falls back to torch. Generation spends one prefill and
    one call here per token, so the fallback costs almost nothing.
    """

    def __init__(self, fallback: torch.nn.Module):
        super().__init__()
        self.design = NpuDesign("lm_head")
        self.fallback = fallback
        self.n_rows = self.design.meta["n_rows"]
        self.k = self.design.meta["k"]
        self.npu_calls = 0
        self.cpu_calls = 0

    def load_weights(self, raw: bytes) -> None:
        self.design.upload(0, np.frombuffer(raw, dtype=np.uint8))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.shape[:-1].numel() != 1:
            self.cpu_calls += 1
            return self.fallback(h)

        t = time.perf_counter()
        x = h.detach().to(torch.float32).reshape(-1).numpy().astype(bfloat16)
        self.design.upload(1, x)
        self.design.run()
        y = self.design.download(2, np.float32, self.n_rows)
        self.design.seconds += time.perf_counter() - t
        self.design.calls += 1
        self.npu_calls += 1
        return torch.from_numpy(y.copy()).reshape(*h.shape[:-1], -1).to(h.dtype)


def attach_lm_head(model, f) -> NpuLmHead:
    """Swap the model's lm_head for the NPU one and upload its weights."""
    e = f.header["lm_head.weight"]
    first, last = e["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + first)
        raw = fh.read(last - first)
    mod = NpuLmHead(model.lm_head)
    mod.load_weights(raw)
    model.lm_head = mod
    return mod
