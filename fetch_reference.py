"""Fetch the upstream correctness reference, checking the recorded source hashes.

This setup step requires the network, not PyTorch or an OKC account. Subsequent
benchmark runs use the verified local files without a network connection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
SOURCE = "https://openkernelchallenge.dev/api/v1/problems/muse_bf16_proj"


def main():
    report = json.loads((HERE / "benchmark-results.json").read_text(encoding="utf-8"))
    expected = {path.replace("\\", "/"): digest for path, digest in report["source_sha256"].items()}
    missing = []
    for name in ("reference.py", "checks.py"):
        target = HERE / "reference" / name
        digest = expected[f"reference/{name}"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"{target.name} differs from the recorded reference. Refusing to overwrite it.")
        else:
            missing.append((name, target, digest))
    if not missing:
        print("Both local reference files match the recorded SHA-256 hashes.")
        return
    request = Request(SOURCE, headers={"User-Agent": "Muse-Sputter-reference-setup/1.0"})
    with urlopen(request, timeout=30) as response:
        files = json.load(response)["files"]
    downloads = []
    for name, target, expected_digest in missing:
        content = files[name].encode("utf-8")
        if hashlib.sha256(content).hexdigest() != expected_digest:
            raise RuntimeError(f"Upstream {name} has changed since the recorded benchmark; no files were written.")
        downloads.append((target, content))
    for target, content in downloads:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        print(f"Fetched and SHA-256 verified reference/{target.name}")


if __name__ == "__main__":
    main()
