r"""Pull single tensors out of a remote `.q4nx` without downloading the model.

`model.q4nx` for Qwen3.5-0.8B-NPU2 is 1.1 GB, but it is a safetensors container:
the first 64 KB holds the whole header (40,888 B of JSON, 357 entries), and any
one tensor is then a second HTTP range request. Everything this repo needs is a
few KB.

    python tools\fetch_q4nx.py --list
    python tools\fetch_q4nx.py model.layers.0.mlp.up_proj.weight --tiles 1
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from q4nx import Q4_TILE_BYTES, read_header  # noqa: E402

REPO = "FastFlowLM/Qwen3.5-0.8B-NPU2"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
FIXTURES = Path(__file__).parent.parent / "designs" / "q4nx_unpack" / "fixtures"
HEADER_PROBE = 65536


def _get(url: str, first: int, last: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_header(file: str = "model.q4nx") -> tuple[dict, int]:
    """Returns (header, data_start). data_start is 8 + len(header JSON)."""
    blob = _get(f"{BASE}/{file}", 0, HEADER_PROBE - 1)
    header = read_header(io.BytesIO(blob))
    return header, 8 + int.from_bytes(blob[:8], "little")


def fetch_tensor(name: str, nbytes: int | None = None, file: str = "model.q4nx"):
    header, data_start = fetch_header(file)
    if name not in header:
        raise KeyError(f"{name!r} not in {file}")
    entry = header[name]
    first, last = entry["data_offsets"]
    if nbytes is not None:
        last = min(last, first + nbytes)
    raw = _get(f"{BASE}/{file}", data_start + first, data_start + last - 1)
    return raw, entry


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="model.layers.0.mlp.up_proj.weight")
    ap.add_argument("--file", default="model.q4nx")
    ap.add_argument("--tiles", type=int, default=1, help="q4 tiles to fetch")
    ap.add_argument("--list", action="store_true", help="print the header and stop")
    ap.add_argument("--force", action="store_true", help="refetch even if cached")
    a = ap.parse_args(argv[1:])

    if a.list:
        header, _ = fetch_header(a.file)
        for k, v in sorted(header.items()):
            if k == "__metadata__":
                continue
            o = v["data_offsets"]
            print(f"{k:70s} {v['dtype']:5s} {str(v['shape']):22s} {o[1] - o[0]}")
        return 0

    FIXTURES.mkdir(parents=True, exist_ok=True)
    out = FIXTURES / f"{a.name}.{a.tiles}tile.bin"
    meta = out.with_suffix(".json")
    if out.exists() and meta.exists() and not a.force:
        print(f"cached  {out.name}  ({out.stat().st_size} B)")
        return 0

    raw, entry = fetch_tensor(a.name, a.tiles * Q4_TILE_BYTES, a.file)
    out.write_bytes(raw)
    meta.write_text(
        json.dumps({"repo": REPO, "file": a.file, "tensor": a.name, **entry}, indent=2)
    )
    print(f"fetched {out.name}  ({len(raw)} B)  from {a.name} {entry['shape']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
