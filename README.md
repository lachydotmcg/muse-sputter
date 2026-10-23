# Muse Sputter

**Every FLOP deserves a work-life balance.**

The Open Kernel Challenge asks for faster Muse Spark Open kernels. This one goes
the other way: correct dense projections through thousands of tiny matrix
multiplies, with a measured **143.8x slowdown overall versus ordinary PyTorch**
on an **RTX 3060 (8 GB)**.

Correct answer. Eventually.

## The receipts

Recorded on 23 October 2026 in Sydney: 30 published `muse_bf16_proj` shapes,
31 paired samples per shape, **930 pairs / 1,860 measured GPU executions**.
The overall slowdown is **143.79x**, with a within-run 95% bootstrap interval
of **137.55-152.32x**.

| Load | Slowdown versus PyTorch |
|---|---:|
| Decode, 1 request | 65.60x |
| Decode, 16 requests | 89.33x |
| Decode, 32 requests | 385.09x |
| Verify, 1 request | 92.94x |
| Verify, 16 requests | 286.70x |
| Verify, 32 requests | 146.95x |
| **Geometric mean across loads** | **143.79x** |

All 30 benchmark shapes passed correctness checks, including checks after
changing the replay input. The nine published stress cases and two extra
strided-input / partial-tile cases also passed.

The comparison is ordinary PyTorch on the same local GPU. It measures dense
projection kernels; the whole model, SGLang and an H200 were not benchmarked.
This is an independent local parody, with no official challenge submission or
affiliation.

- [Raw paired timings and correctness results](benchmark-results.json)
- [Per-shape timing table](benchmark-results.csv)
- [Console output](benchmark-run.log)
- [Independent arithmetic and source-hash audit](verification.json)
- [Second timing method's measurements](timing-crosscheck.json)
- [Exact installed package versions](environment.lock.txt)

## How to make a GPU sad

The normal implementation computes `x @ W.T` in one call. Muse Sputter splits
output columns into groups of **16** and the reduction dimension into chunks of
**256**, then processes each piece in sequence. The QKV projection alone becomes
**4,608 tiny matrix multiplies** instead of one.

Each mathematical product is computed once. Partial sums accumulate in FP32,
with one final cast to BF16 for BF16 projections. Router weights and output stay
FP32. There are no sleeps, repeated full projections, cached answers or timing
hooks in [the kernel](kernel.py).

## Check the recorded result

Python 3.14 is sufficient for this audit. No GPU or PyTorch installation is
needed:

```sh
git clone https://github.com/lachydotmcg/muse-sputter.git
cd muse-sputter
python fetch_reference.py
python verify_results.py
```

The setup script downloads the challenge's two correctness-reference files and
verifies them against the original benchmark's SHA-256 hashes. They are fetched
separately from the public API rather than redistributed here. No OKC login is
required. Once fetched, benchmark runs work offline.

The automated check audits the recorded data and hashes. GPU performance is
reproduced by running the benchmark on CUDA hardware.

## Run the benchmark

On a CUDA-capable Windows machine, from the repository root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe fetch_reference.py
.\.venv\Scripts\python.exe benchmark.py --output local-results
```

The `local-results` prefix preserves the recorded evidence. See
`python benchmark.py --help` for shape filters, sample counts and tile sizes.
The dependencies pin PyTorch 2.10.0+cu128 and NumPy 2.5.4; the recorded run used
CUDA 12.8, Python 3.14.4 and Windows 11.

## Timing and aggregation

- CUDA timing-event nodes are captured inside both implementations' CUDA graphs,
  excluding Python graph-construction overhead. No fixed cost is subtracted.
- Execution order alternates. A 128 MiB write flush runs before each side,
  outside its timed interval. This differs from the official judge's read flush.
- Four seeded inputs rotate across paired samples; each pair uses the same
  input for both implementations. Correctness checks use the downloaded FP64
  reference, its tolerances, probe rows and outlier distributions.
- TF32 and BF16 reduced-precision reduction are disabled for both sides.
- For each load, sum each shape's median time multiplied by its challenge
  weight. Divide the parody sum by the PyTorch sum. The headline is the
  geometric mean of those six load ratios. It is an aggregate, not a claim
  that every shape is 143.79x slower.
- Tile sizes were chosen during exploratory runs before the final 31-sample
  measurement. All raw pairs, seeds, hardware details and core source hashes
  are included in the JSON.

A separate check used ordinary CUDA events enclosing a graph replay. Its median
times agreed with the captured-event method within 1% on both implementations
at the 32-token QKV shape.

The RTX 3060 is a shared Windows desktop GPU with uncontrolled clocks and
background activity. Bootstrap intervals describe variation within this run;
separate runs may vary.

## Easter eggs

| Egg | Where it lives |
|---|---|
| **Muse Sputter** | Project name and benchmark banner |
| "Every FLOP deserves a work-life balance." | README and kernel docstring |
| "Tensor cores may only enter one at a time." | The small-tile loop |
| `bureaucracy_mode` | Benchmark banner and JSON |
| "Correct answer. Eventually." | Successful benchmark completion |

## Source and license

The [challenge problem](https://openkernelchallenge.dev/problems/muse_bf16_proj)
and [public API](https://openkernelchallenge.dev/api/v1/problems/muse_bf16_proj)
provide the workload shapes and correctness-reference files. See also the
[challenge's agent instructions](https://openkernelchallenge.dev/llms.txt).
The upstream reference files belong to their authors and are fetched separately.
The original project files in this repository are [MIT licensed](LICENSE).
