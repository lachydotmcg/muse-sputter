"""Audit the saved measurements and source hashes without requiring PyTorch."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent
    report_path = directory / "benchmark-results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete", "Benchmark is incomplete."
    assert "RTX 3060" in report["hardware"]["gpu"]
    assert not report["method"]["stress_tests_skipped"]
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    expected_cases = {(c["m"], c["layer"], c["load"]) for c in cases}
    actual_cases = {(c["m"], c["layer"], c["load"]) for c in report["results"]}
    assert actual_cases == expected_cases and len(report["results"]) == 30
    for relative, digest in report["source_sha256"].items():
        assert hashlib.sha256((directory / relative.replace("\\", "/")).read_bytes()).hexdigest() == digest, relative
    stress = [v for v in report["validation"] if v["kind"] == "published stress shape"]
    tails = [v for v in report["validation"] if v["kind"] == "partial tiles and strided input"]
    assert len(stress) == 9 and len(tails) == 2
    assert all(v["baseline"]["passed"] and v["parody"]["passed"] for v in stress)
    assert all(v["check"]["passed"] for v in tails)
    groups = {}
    for row in report["results"]:
        samples = row["paired_samples"]
        assert len(samples) == report["method"]["paired_samples_per_shape"]
        assert row["baseline_fresh_input_check"]["passed"]
        assert row["parody_fresh_input_check"]["passed"]
        for sample in samples:
            assert sample["baseline_ms"] > 0 and sample["parody_ms"] > 0
            assert math.isclose(sample["slowdown"], sample["parody_ms"] / sample["baseline_ms"], rel_tol=1e-12)
        fast = statistics.median(p["baseline_ms"] for p in samples)
        slow = statistics.median(p["parody_ms"] for p in samples)
        assert math.isclose(fast, row["baseline_median_ms"], rel_tol=1e-12)
        assert math.isclose(slow, row["parody_median_ms"], rel_tol=1e-12)
        group = groups.setdefault(row["load"], [0.0, 0.0])
        group[0] += row["weight"] * fast
        group[1] += row["weight"] * slow
    slowdown = math.exp(statistics.mean(math.log(v[1] / v[0]) for v in groups.values()))
    assert math.isclose(slowdown, report["summary"]["geometric_mean_load_slowdown"], rel_tol=1e-12)
    verification = {
        "passed": True,
        "source_hashes_match": True,
        "benchmark_shapes": len(report["results"]),
        "published_stress_cases": len(stress),
        "additional_edge_cases": len(tails),
        "paired_samples": sum(len(r["paired_samples"]) for r in report["results"]),
        "independently_recomputed_slowdown": slowdown,
        "benchmark_results_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "scope": "Saved-data arithmetic and provenance audit; GPU timings are measured by benchmark.py.",
    }
    (directory / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
