#!/usr/bin/env python3
"""
Benchmark scipy.fft complex128 throughput for 1D/2D/3D forward and inverse FFTs.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

import numpy as np

try:
    from scipy import fft as scipy_fft
except Exception as exc:  # pragma: no cover - runtime dependency check
    print(f"error=failed_to_import_scipy:{exc}", file=sys.stderr)
    raise


LCG_A = 6364136223846793005
LCG_C = 1442695040888963407
MASK64 = (1 << 64) - 1
U53_SCALE = float(1 << 53)


def lcg_unit_values(seed: int, count: int) -> np.ndarray:
    state = int(seed) & MASK64
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        state = (state * LCG_A + LCG_C) & MASK64
        out[i] = float(state >> 11) / U53_SCALE
    return out


def build_flat_input(total_len: int, input_kind: str, seed: int) -> np.ndarray:
    idx = np.arange(total_len, dtype=np.float64)
    t = idx / float(total_len)

    if input_kind == "impulse":
        out = np.zeros(total_len, dtype=np.complex128)
        out[0] = 1.0 + 0.0j
        return out
    if input_kind == "ramp":
        return t + 1j * (1.0 - t)
    if input_kind == "multitone":
        re = np.sin(2.0 * math.pi * 3.0 * t) + 0.5 * np.sin(
            2.0 * math.pi * 11.0 * t + 0.3
        )
        im = np.cos(2.0 * math.pi * 5.0 * t) - 0.25 * np.sin(2.0 * math.pi * 9.0 * t)
        return re + 1j * im
    if input_kind == "chirp":
        phase = math.pi * idx * idx / float(total_len)
        return np.cos(phase) + 1j * np.sin(phase)
    if input_kind == "random":
        vals = lcg_unit_values(seed, 2 * total_len)
        re = 2.0 * vals[0::2] - 1.0
        im = 2.0 * vals[1::2] - 1.0
        return re + 1j * im
    raise ValueError(f"unsupported input_kind: {input_kind}")


def apply_op(op: str, data: np.ndarray, workers: int) -> np.ndarray:
    if op == "forward1d":
        return scipy_fft.fft(data, overwrite_x=True, workers=workers)
    if op == "inverse1d":
        return scipy_fft.ifft(data, overwrite_x=True, workers=workers)
    if op == "forward2d":
        return scipy_fft.fft2(data, overwrite_x=True, workers=workers)
    if op == "inverse2d":
        return scipy_fft.ifft2(data, overwrite_x=True, workers=workers)
    if op == "forward3d":
        return scipy_fft.fftn(data, overwrite_x=True, workers=workers)
    if op == "inverse3d":
        return scipy_fft.ifftn(data, overwrite_x=True, workers=workers)
    raise ValueError(f"unsupported op: {op}")


def prepare_data(args: argparse.Namespace) -> tuple[np.ndarray, int]:
    op = args.op
    if op in ("forward1d", "inverse1d"):
        if args.n is None:
            raise ValueError("--n is required for 1D operations")
        if args.batch <= 0:
            raise ValueError("batch must be > 0")
        total_len = args.n * args.batch
        flat = build_flat_input(total_len, args.input_kind, args.seed)
        if args.batch == 1:
            data = flat
        else:
            data = flat.reshape((args.batch, args.n))
        return data.astype(np.complex128, copy=False), total_len

    if op in ("forward2d", "inverse2d"):
        if args.width is None or args.height is None:
            raise ValueError("--width and --height are required for 2D operations")
        total_len = args.width * args.height
        flat = build_flat_input(total_len, args.input_kind, args.seed)
        data = flat.reshape((args.height, args.width))
        return data.astype(np.complex128, copy=False), total_len

    if op in ("forward3d", "inverse3d"):
        if args.width is None or args.height is None or args.depth is None:
            raise ValueError("--width, --height, and --depth are required for 3D operations")
        total_len = args.width * args.height * args.depth
        flat = build_flat_input(total_len, args.input_kind, args.seed)
        data = flat.reshape((args.depth, args.height, args.width))
        return data.astype(np.complex128, copy=False), total_len

    raise ValueError(f"unsupported op: {op}")


def run_benchmark(args: argparse.Namespace) -> tuple[float, float]:
    baseline, _ = prepare_data(args)
    work = np.array(baseline, copy=True)

    # Warm-up to amortize setup and page-fault effects before sampling.
    for _ in range(3):
        np.copyto(work, baseline)
        work = apply_op(args.op, work, args.workers)

    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        for _ in range(args.iterations):
            np.copyto(work, baseline)
            work = apply_op(args.op, work, args.workers)
        elapsed = time.perf_counter() - start
        samples.append(elapsed * 1e6 / args.iterations)

    median_us_per_fft = statistics.median(samples)
    total_seconds = median_us_per_fft * args.iterations / 1e6
    return total_seconds, median_us_per_fft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        type=str,
        required=True,
        choices=[
            "forward1d",
            "inverse1d",
            "forward2d",
            "inverse2d",
            "forward3d",
            "inverse3d",
        ],
    )
    parser.add_argument("--n", type=int)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--input-kind",
        type=str,
        required=True,
        choices=["impulse", "ramp", "multitone", "chirp", "random"],
    )
    parser.add_argument("--seed", type=int, default=0x12345678)
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("iterations must be > 0")
    if args.samples <= 0:
        raise ValueError("samples must be > 0")
    if args.workers == 0:
        raise ValueError("workers must not be 0")
    if args.n is not None and args.n <= 0:
        raise ValueError("n must be > 0")
    if args.batch <= 0:
        raise ValueError("batch must be > 0")
    if args.width is not None and args.width <= 0:
        raise ValueError("width must be > 0")
    if args.height is not None and args.height <= 0:
        raise ValueError("height must be > 0")
    if args.depth is not None and args.depth <= 0:
        raise ValueError("depth must be > 0")

    elapsed, us_per_fft = run_benchmark(args)

    print("backend=scipy.fft")
    print(f"op={args.op}")
    if args.n is not None:
        print(f"n={args.n}")
        print(f"batch={args.batch}")
    if args.width is not None:
        print(f"width={args.width}")
    if args.height is not None:
        print(f"height={args.height}")
    if args.depth is not None:
        print(f"depth={args.depth}")
    print(f"iterations={args.iterations}")
    print(f"workers={args.workers}")
    print(f"input_kind={args.input_kind}")
    print(f"total_seconds={elapsed:.12f}")
    print(f"us_per_fft={us_per_fft:.6f}")


if __name__ == "__main__":
    main()
