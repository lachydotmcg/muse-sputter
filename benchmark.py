"""Reproducible local GPU benchmark for Muse Sputter.

Both implementations use identical BF16 inputs and weights. CUDA event nodes
inside each CUDA graph measure GPU work, excluding Python submission time.
The slow kernel is never allowed to alter the timer or benchmark inputs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import torch

from kernel import MODE_NAME, SUCCESS_MESSAGE, baseline_fn, kernel_fn, prepare


HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / "reference"
sys.path.insert(0, str(REFERENCE_DIR))
import reference
from checks import compare


STRESS_CASES = [
    (1, "qkv_gate", "normal"),
    (7, "pooler", "normal"),
    (33, "o", "normal"),
    (130, "unpooler", "normal"),
    (5, "unpooler", "normal"),
    (8, "router", "normal"),
    (9, "router", "outliers"),
    (512, "qkv_gate", "outliers"),
    (1000, "unpooler", "outliers"),
]


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and NVIDIA GPU are required.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This benchmark requires native BF16 support.")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def check(data, output, weights):
    """Use the downloaded reference and its published tolerances and probes."""
    config = data["config"]
    tolerance = reference.ROUTER_TOL if config["layer"] == "router" else reference.REL_L2_TOL
    expected = reference.ref_kernel(data, weights)
    ok, message = compare(output, expected, tolerance)
    relative_l2 = float((output.double() - expected.double()).norm() / expected.double().norm().clamp(min=1e-30))
    probes = reference.probe_rows(data["x"])
    probe_error = 0.0
    if len(probes):
        error = (output[probes].double() - expected[probes].double()).norm(dim=-1)
        error /= expected[probes].double().norm(dim=-1).clamp(min=1e-30)
        probe_error = float(error.max())
        if probe_error > reference.PROBE_TOL:
            ok = False
            message += f"; probe error {probe_error:.3e} > {reference.PROBE_TOL}"
    if not ok:
        raise AssertionError(f"{config}: {message}")
    return {"passed": True, "relative_l2": relative_l2, "probe_rows": len(probes),
            "max_probe_relative_l2": probe_error, "message": message}


class Captured:
    def __init__(self, fn, warmup):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(self.graph):
            self.start.record()
            self.output = fn()
            self.end.record()
        for _ in range(warmup):
            self.graph.replay()
        torch.cuda.synchronize()

    def measure(self, cache_buffer, fill):
        # Flush before BOTH implementations, outside their recorded intervals.
        cache_buffer.fill_(fill)
        self.graph.replay()
        self.end.synchronize()
        return self.start.elapsed_time(self.end)


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def bootstrap_median_interval(values, seed=5151):
    rng = random.Random(seed)
    estimates = [statistics.median(rng.choices(values, k=len(values))) for _ in range(2000)]
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def check_tail_tiles(state):
    """Exercise shapes that are not divisible by either tile dimension."""
    results = []
    for layer, dtype in [("qkv_gate", torch.bfloat16), ("router", torch.float32)]:
        generator = torch.Generator(device="cuda").manual_seed(19517)
        k, n = state.reduction_chunk + 13, state.output_columns + 11
        # Strided inputs also catch accidental assumptions about contiguity.
        x = torch.randn((3, 2 * k), generator=generator, device="cuda").to(torch.bfloat16)[:, ::2]
        w = torch.randn((n, 2 * k), generator=generator, device="cuda").to(dtype)[:, ::2]
        test_state = prepare({layer: w}, output_columns=state.output_columns,
                             reduction_chunk=state.reduction_chunk)
        data = {"x": x, "config": {"m": 3, "n": n, "k": k, "layer": layer}}
        results.append({"kind": "partial tiles and strided input", "layer": layer,
                        "check": check(data, kernel_fn(data, test_state), {layer: w})})
    return results


def hardware_info():
    properties = torch.cuda.get_device_properties(0)
    try:
        telemetry = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw",
             "--format=csv,noheader"], text=True, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError):
        telemetry = "unavailable"
    return {"gpu": properties.name, "compute_capability": list(torch.cuda.get_device_capability()),
            "total_vram_bytes": properties.total_memory, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
            "os": platform.platform(), "nvidia_smi": telemetry}


def source_hashes():
    files = [HERE / "kernel.py", HERE / "benchmark.py", HERE / "cases.json",
             REFERENCE_DIR / "reference.py", REFERENCE_DIR / "checks.py"]
    return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def summarize(results):
    groups = {}
    for row in results:
        group = groups.setdefault(row["load"], {"baseline_ms": 0.0, "parody_ms": 0.0})
        group["baseline_ms"] += row["baseline_median_ms"] * row["weight"]
        group["parody_ms"] += row["parody_median_ms"] * row["weight"]
    for group in groups.values():
        group["slowdown"] = group["parody_ms"] / group["baseline_ms"]
    aggregate = math.exp(statistics.mean(math.log(g["slowdown"]) for g in groups.values()))
    # Bootstrap paired samples within each shape, then apply the same load formula.
    rng = random.Random(5151)
    estimates = []
    for _ in range(1000):
        sampled_groups = {}
        for row in results:
            samples = row["paired_samples"]
            resampled = rng.choices(samples, k=len(samples))
            group = sampled_groups.setdefault(row["load"], [0.0, 0.0])
            group[0] += statistics.median(p["baseline_ms"] for p in resampled) * row["weight"]
            group[1] += statistics.median(p["parody_ms"] for p in resampled) * row["weight"]
        estimates.append(math.exp(statistics.mean(math.log(v[1] / v[0]) for v in sampled_groups.values())))
    return {"geometric_mean_load_slowdown": aggregate,
            "bootstrap_95_percent_interval": [quantile(estimates, 0.025), quantile(estimates, 0.975)],
            "loads": groups,
            "formula": "For each load: sum(weight * parody median time) / sum(weight * baseline median time). Geometric mean across loads."}


def save_report(report, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["results"]:
        return
    fields = ["load", "layer", "m", "n", "k", "weight", "baseline_median_ms", "parody_median_ms",
              "median_paired_slowdown", "baseline_iqr_ms", "parody_iqr_ms", "ratio_bootstrap_95_percent_interval"]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=int, default=16)
    parser.add_argument("--reduction-chunk", type=int, default=256)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--layer", choices=list(reference.LAYERS))
    parser.add_argument("--m", type=int)
    parser.add_argument("--skip-stress", action="store_true", help="For exploratory runs only.")
    parser.add_argument("--output", type=Path, default=HERE / "benchmark-results")
    args = parser.parse_args()
    if args.samples < 7 or args.warmup < 1:
        parser.error("Use at least 7 paired samples and one warmup replay.")
    configure_cuda()
    hardware = hardware_info()
    print(f"Muse Sputter | {MODE_NAME} | {hardware['gpu']}", flush=True)
    print(f"PyTorch {hardware['torch']}; CUDA {hardware['cuda_runtime']}; Python {hardware['python']}", flush=True)
    print(f"Tiles: {args.columns} output columns x {args.reduction_chunk} reduction entries.", flush=True)
    try:
        now = datetime.now(ZoneInfo("Australia/Sydney")).isoformat()
    except ZoneInfoNotFoundError:
        now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=11))).isoformat()
    report = {
        "title": "Muse Sputter", "mode": MODE_NAME, "started_at_sydney": now,
        "status": "running", "hardware": hardware,
        "comparison": "Local PyTorch torch.mm vs deliberately fragmented PyTorch projection on the same RTX 3060. This is not SGLang, H200, a whole-model benchmark, or an official OKC result.",
        "source": "https://openkernelchallenge.dev/api/v1/problems/muse_bf16_proj",
        "method": {"timing": "CUDA event nodes captured inside each CUDA graph; no fixed-overhead subtraction",
                   "paired_samples_per_shape": args.samples, "warmup_replays": args.warmup,
                   "ordering": "baseline first on even pairs, parody first on odd pairs",
                   "fresh_inputs": "4 distinct seeded input tensors rotate across pairs; the same tensor is used for both sides of a pair",
                   "cache": "128 MiB write flush before each side, outside timing; not the judge's exact read-flush protocol",
                   "tf32": False, "bf16_reduced_precision_reduction": False,
                   "clock_control": "Unchanged; shared Windows desktop GPU; no exclusive-GPU or clock-lock claim",
                   "stress_tests_skipped": args.skip_stress},
        "tiles": {"output_columns": args.columns, "reduction_chunk": args.reduction_chunk},
        "source_sha256": source_hashes(), "validation": [], "results": [],
    }
    weights = reference.generate_weights(reference.WEIGHTS_SEED)
    state = prepare(weights, output_columns=args.columns, reduction_chunk=args.reduction_chunk)
    report["validation"].extend(check_tail_tiles(state))
    if not args.skip_stress:
        for index, (m, layer, dist) in enumerate(STRESS_CASES):
            print(f"Correctness {index+1}/9: m={m} {layer} {dist} ...", flush=True)
            data = reference.generate_input(m, layer, 9001 + index, dist=dist)
            baseline_check = check(data, baseline_fn(data, state), weights)
            parody_check = check(data, kernel_fn(data, state), weights)
            report["validation"].append({"kind": "published stress shape", "m": m,
                                         "layer": layer, "distribution": dist,
                                         "baseline": baseline_check, "parody": parody_check})
            print(f"  PASS: error {parody_check['relative_l2']:.3e}", flush=True)
        del data
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    cases = [c for c in cases if (args.layer is None or c["layer"] == args.layer)
             and (args.m is None or c["m"] == args.m)]
    if not cases:
        raise ValueError("No benchmark cases match these filters.")
    cache_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for index, case in enumerate(cases):
        m, layer = case["m"], case["layer"]
        print(f"Benchmark {index+1}/{len(cases)}: {case['load']} / {layer} ...", flush=True)
        data = reference.generate_input(m, layer, case["seed"])
        input_seeds = [case["seed"] + 10000 + bank for bank in range(4)]
        banks = [reference.generate_input(m, layer, seed)["x"] for seed in input_seeds]
        baseline = Captured(lambda: baseline_fn(data, state), args.warmup)
        parody = Captured(lambda: kernel_fn(data, state), args.warmup)
        check(data, baseline.output, weights)
        check(data, parody.output, weights)
        samples = []
        for sample in range(args.samples):
            data["x"].copy_(banks[sample % len(banks)])
            order = [("baseline", baseline), ("parody", parody)]
            if sample % 2:
                order.reverse()
            times = {}
            for name, graph in order:
                times[name + "_ms"] = graph.measure(cache_buffer, (sample + 1) % 127)
            times["slowdown"] = times["parody_ms"] / times["baseline_ms"]
            samples.append(times)
        row = {**case, "n": data["config"]["n"], "k": data["config"]["k"],
               "input_seeds": input_seeds, "paired_samples": samples,
               "baseline_median_ms": statistics.median(p["baseline_ms"] for p in samples),
               "parody_median_ms": statistics.median(p["parody_ms"] for p in samples),
               "median_paired_slowdown": statistics.median(p["slowdown"] for p in samples),
               "baseline_iqr_ms": [quantile([p["baseline_ms"] for p in samples], q) for q in [0.25, 0.75]],
               "parody_iqr_ms": [quantile([p["parody_ms"] for p in samples], q) for q in [0.25, 0.75]],
               "ratio_bootstrap_95_percent_interval": bootstrap_median_interval([p["slowdown"] for p in samples]),
               "baseline_fresh_input_check": check(data, baseline.output, weights),
               "parody_fresh_input_check": check(data, parody.output, weights)}
        report["results"].append(row)
        print(f"  PyTorch {row['baseline_median_ms']:.4f} ms | parody {row['parody_median_ms']:.4f} ms | {row['median_paired_slowdown']:.2f}x slower", flush=True)
        save_report(report, args.output)
        del baseline, parody, banks, data, order, graph
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    report["summary"] = summarize(report["results"])
    report["status"] = "complete"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["ending_hardware"] = hardware_info()
    save_report(report, args.output)
    ratio = report["summary"]["geometric_mean_load_slowdown"]
    interval = report["summary"]["bootstrap_95_percent_interval"]
    print(f"\n{SUCCESS_MESSAGE}", flush=True)
    print(f"Aggregate: {ratio:.2f}x slower; bootstrap 95% interval {interval[0]:.2f}-{interval[1]:.2f}x.", flush=True)
    print(f"Raw paired samples: {args.output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
