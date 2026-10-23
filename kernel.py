"""Muse Sputter: a deliberately inefficient, numerically correct projection.

Every FLOP deserves a work-life balance.

This is a local parody of Open Kernel Challenge's muse_bf16_proj. It performs
the actual multiplication once, in tiny pieces. There are no sleeps, duplicate
matrix multiplies, cached answers, timing hooks, or imports of the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


MODE_NAME = "bureaucracy_mode"
SUCCESS_MESSAGE = "Correct answer. Eventually."


@dataclass(frozen=True)
class State:
    weights: dict[str, torch.Tensor]
    output_columns: int = 16
    reduction_chunk: int = 256


def prepare(
    weights: dict[str, torch.Tensor],
    *,
    output_columns: int = 16,
    reduction_chunk: int = 256,
) -> State:
    """Keep every weight at its original precision; don't inspect activations."""
    if output_columns < 1 or reduction_chunk < 1:
        raise ValueError("Tile dimensions must be positive.")
    return State(weights, output_columns, reduction_chunk)


def baseline_fn(data: dict, state: State) -> torch.Tensor:
    """Ordinary PyTorch GEMM, not an installed SGLang or H200 baseline."""
    config = data["config"]
    weight = state.weights[config["layer"]][: config["n"]]
    x = data["x"]
    if weight.dtype == torch.float32:
        x = x.float()
    return torch.mm(x, weight.t())


def kernel_fn(data: dict, state: State) -> torch.Tensor:
    """Compute x @ W[:n].T through many serial, small matrix multiplies.

    Each input/weight product belongs to exactly one tile. All partial sums
    accumulate in FP32; only the finished BF16 projection is rounded to BF16.
    This accepts arbitrary positive sizes, including partial tiles.
    """
    config = data["config"]
    weight = state.weights[config["layer"]][: config["n"]]
    x = data["x"]
    if weight.dtype == torch.float32:
        x = x.float()
    m, k = x.shape
    n = weight.shape[0]
    output = torch.empty((m, n), dtype=weight.dtype, device=x.device)

    # Tensor cores may only enter one at a time.
    for column in range(0, n, state.output_columns):
        stop = min(column + state.output_columns, n)
        total = torch.zeros((m, stop - column), dtype=torch.float32, device=x.device)
        for start in range(0, k, state.reduction_chunk):
            end = min(start + state.reduction_chunk, k)
            a = x[:, start:end]
            b = weight[column:stop, start:end].t()
            if weight.dtype == torch.bfloat16:
                partial = torch.mm(a, b, out_dtype=torch.float32)
            else:
                partial = torch.mm(a, b)
            total.add_(partial)
        output[:, column:stop].copy_(total)
    return output
