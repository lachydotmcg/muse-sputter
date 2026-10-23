# Muse Sputter

I vibecoded this in 20 minutes. Please do not download this or use it, for your
own sake.

Yes, I'm that bored.

The Open Kernel Challenge wants faster Muse Spark Open kernels. I aimed for
100x slower and got **143.8x slower overall than PyTorch on my RTX 3060**.
I split matrix multiplication into thousands of tiny operations that run one
after another. Every FLOP deserves a work-life balance.

All 30 projection shapes passed correctness checks, along with nine stress cases
and two extra edge cases. The benchmark has 930 paired measurements and a 95%
bootstrap interval of 137.55-152.32x. That's the aggregate slowdown for dense
projections on my desktop GPU, not a whole-model or SGLang result. Separate runs
may vary.

[Raw results](benchmark-results.json) · [Per-shape timings](benchmark-results.csv)

## Check the recorded result

For whatever reason you would want to 😭

Python 3.14 is enough. You don't need a GPU or PyTorch for this part.

```sh
git clone https://github.com/lachydotmcg/muse-sputter.git
cd muse-sputter
python fetch_reference.py
python verify_results.py
```

This fetches the challenge's reference files and checks their hashes, then
audits the saved measurements.

## Run the benchmark

On Windows with a CUDA-capable GPU, from the repo folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe fetch_reference.py
.\.venv\Scripts\python.exe benchmark.py --output local-results
```

Your results go into `local-results` files. Run `python benchmark.py --help` to
change shapes, sample counts or tile sizes.

Correct answer. Eventually.
