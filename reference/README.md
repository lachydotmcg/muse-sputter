# Upstream correctness reference

Run `python fetch_reference.py` from the repository root to download
`reference.py` and `checks.py` from the Open Kernel Challenge's public API.
They are the challenge organizers' code and are not redistributed in this repo.

The setup script checks their SHA-256 hashes against the sources used for the
recorded 23 October 2026 benchmark. It refuses changed upstream code and does not
overwrite modified local files. The benchmark imports the reference; the
projection kernel does not.

Source: https://openkernelchallenge.dev/api/v1/problems/muse_bf16_proj
