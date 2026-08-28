r"""Chat with Qwen3.5-0.8B, weights read out of FastFlowLM's q4nx file.

By default the model's largest projection -- lm_head, 34% of the arithmetic in a
token -- runs on the NPU and everything else runs in torch on the CPU. `--cpu`
turns the NPU off, so the two are the same harness and the difference is the
array.

    ..\NpuEmbeddings\.venv-ref\Scripts\python.exe chat.py "why is the sky blue?"
    ..\NpuEmbeddings\.venv-ref\Scripts\python.exe chat.py --cpu "..."
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "reference"))
sys.path.insert(0, str(Path(__file__).parent / "tools"))
from from_q4nx import MODEL_DIR, build_model, load_tokenizer  # noqa: E402

MAX_NEW = 128


def main(argv: list[str]) -> int:
    import torch

    use_npu = "--cpu" not in argv
    argv = [a for a in argv if a != "--cpu"]

    t0 = time.time()
    model = build_model()
    tok = load_tokenizer()

    head = None
    if use_npu:
        import q4nx
        from npu import attach_lm_head

        head = attach_lm_head(model, q4nx.Q4NX(MODEL_DIR / "model.q4nx"))
        print(f"npu      lm_head on the array, {head.n_rows} x "
              f"{head.k}, 270 MB, 8 cores")
    else:
        print("cpu      everything in torch")
    print(f"ready    {time.time() - t0:.1f}s\n")

    one_shot = " ".join(argv[1:]).strip()
    history: list[dict] = []

    while True:
        if one_shot:
            prompt = one_shot
            print(f"> {prompt}")
        else:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                prompt = ""
            if not prompt or prompt in {"exit", "quit"}:
                if head is not None:
                    head.design.close()
                return 0

        history.append({"role": "user", "content": prompt})
        ids = tok.apply_chat_template(
            history, add_generation_prompt=True, return_tensors="pt",
            return_dict=False, enable_thinking=False,
        )

        if head is not None:
            head.design.seconds = head.design.calls = head.npu_calls = 0
        t = time.time()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        new = out[0, ids.shape[-1]:]
        text = tok.decode(new, skip_special_tokens=True)
        dt = time.time() - t

        print(text)
        line = f"[{len(new)} tokens, {dt:.1f}s, {len(new) / dt:.2f} tok/s"
        if head is not None:
            per = head.design.seconds / max(head.design.calls, 1) * 1e3
            line += (f"; lm_head on npu {head.npu_calls}x at {per:.1f} ms, "
                     f"{100 * head.design.seconds / dt:.0f}% of wall")
        print(f"\n{line}]\n")
        history.append({"role": "assistant", "content": text})

        if one_shot:
            if head is not None:
                head.design.close()
            return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
