# `testfftr`

Minimal macOS-only FFT crate for `Complex<f64>` backed by Apple Accelerate.

Public API:

- `fft` / `ifft`
- `fft2` / `ifft2`
- `fft3` / `ifft3`
- optional `*_with_workers` variants behind `parallel`
- `AccelerateFFT` for explicit reusable 1D plans

## Scope

This crate is a thin Accelerate wrapper, not a general FFT planner.

Backend strategy:

- radix-2 1D lengths: `vDSP_fft_zipD`
- other supported 1D lengths: `vDSP_DFT_*`
- 2D and 3D: internal composition from reusable 1D plans

Current multidimensional executor split:

- 2D: transpose-based worker path
- 3D: blocked scratch path for non-contiguous axes

## Requirements

- macOS
- Apple Accelerate framework
- Rust 2021

Non-macOS targets fail at compile time.

## Features

- default: minimal dependency footprint
- `parallel`: Rayon-backed worker-controlled execution

```toml
[dependencies]
testfftr = { path = ".", features = ["parallel"] }
```

## API

### Simple API

```rust
use num_complex::Complex;
use testfftr::{fft, fft2, fft3, ifft, ifft2, ifft3};

let mut a = vec![Complex::new(1.0, 0.0); 1024];
fft(&mut a);
ifft(&mut a);

let mut b = vec![Complex::new(0.0, 0.0); 64 * 64];
fft2(&mut b, 64, 64);
ifft2(&mut b, 64, 64);

let mut c = vec![Complex::new(0.0, 0.0); 32 * 16 * 8];
fft3(&mut c, 32, 16, 8);
ifft3(&mut c, 32, 16, 8);
```

Inverse transforms are unnormalized:

- `ifft(fft(x)) ~= N * x`
- `ifft2(fft2(x)) ~= (width * height) * x`
- `ifft3(fft3(x)) ~= (width * height * depth) * x`

### Explicit 1D plan reuse

```rust
use num_complex::Complex;
use testfftr::AccelerateFFT;

let mut plan = AccelerateFFT::new(1024);
let mut data = vec![Complex::new(0.0, 0.0); 1024];

plan.forward(&mut data);
plan.inverse(&mut data);
```

Use `AccelerateFFT` when repeatedly transforming one 1D length.

### Worker-controlled API

Available with `--features parallel`:

- `fft_with_workers` / `ifft_with_workers`
- `fft2_with_workers` / `ifft2_with_workers`
- `fft3_with_workers` / `ifft3_with_workers`
- `fft_batch_with_workers` / `ifft_batch_with_workers`

Worker semantics:

- `workers > 0`: exact requested count
- `workers == -1`: all logical cores
- `workers < -1`: `available_cores - (abs(workers) - 1)`

Parallelism is over independent transforms / lines. There is no intra-kernel multithreading inside a single 1D FFT.

## Design Notes

Accelerate expects split-complex storage:

- real values in one contiguous array
- imaginary values in another contiguous array

The Rust API uses interleaved `Complex<f64>`, so every public execution does:

1. interleaved -> split conversion
2. Accelerate execution
3. split -> interleaved conversion

The important optimization is plan-local buffer reuse. Repeated execution avoids heap allocation.

## Build And Test

```bash
cargo test
cargo test --features parallel
```

Full SciPy comparison matrix:

```bash
cargo test --release --features parallel --test benchmark_compare -- --ignored --nocapture
```

Focused large-workload benchmark used for the table below:

```bash
cargo test --release --features parallel --test benchmark_focus -- --ignored --nocapture
```

SciPy driver:

`benchmarks/scipy_fft_bench.py`

## Focused Benchmark Snapshot

Environment:

- date: March 13, 2026
- command: `cargo test --release --features parallel --test benchmark_focus -- --ignored --nocapture`
- methodology: median of 9 samples, reset from baseline before each measured FFT

Interpretation:

- ratio = `scipy_us / accelerate_us`
- `> 1.0x` means this crate is faster
- `< 1.0x` means SciPy is faster

| Workload | Input | Workers | Accel Fwd us/op | SciPy Fwd us/op | Fwd Ratio | Accel Inv us/op | SciPy Inv us/op | Inv Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1D regular | `multitone` | `2` | 13591.271 | 16812.021 | 1.237x | 13409.785 | 15233.792 | 1.136x |
| 1D regular full-core | `multitone` | `-1` | 13376.965 | 14747.896 | 1.102x | 13355.576 | 16275.174 | 1.219x |
| 2D regular | `multitone` | `2` | 5164.281 | 4174.552 | 0.808x | 5164.583 | 4507.094 | 0.873x |
| 2D regular full-core | `multitone` | `-1` | 3907.333 | 2410.291 | 0.617x | 3911.562 | 2367.896 | 0.605x |
| 2D challenging | `random` | `2` | 1620.911 | 1124.724 | 0.694x | 1578.136 | 1122.094 | 0.711x |
| 2D challenging full-core | `random` | `-1` | 1367.609 | 634.010 | 0.464x | 1372.943 | 636.172 | 0.463x |
| 3D regular | `multitone` | `2` | 11962.695 | 4393.833 | 0.367x | 11146.861 | 4537.611 | 0.407x |
| 3D regular full-core | `multitone` | `-1` | 6009.778 | 2260.208 | 0.376x | 6034.528 | 2232.347 | 0.370x |
| 3D challenging | `random` | `2` | 1794.719 | 742.260 | 0.414x | 1675.521 | 732.312 | 0.437x |
| 3D challenging full-core | `random` | `-1` | 1007.490 | 460.885 | 0.457x | 1009.031 | 526.541 | 0.522x |

## Current Performance Reading

- 1D is in reasonable shape and often ahead of SciPy on large power-of-two workloads.
- 2D and 3D are not yet competitive on the focused large-workload benchmark.
- The main cost is multidimensional data movement, not the radix-2 1D kernel itself.
- Small full-matrix benchmark cells can still be noisy. Use `benchmark_focus` for README-level claims.

## Known Limitations

- macOS only
- complex `f64` only
- no real-to-complex or complex-to-real API
- no public reusable 2D / 3D plan types
- non-radix-2 support depends on what Accelerate can instantiate on the current machine
- 2D / 3D throughput is currently limited by internal layout movement and transpose/gather/scatter overhead

## Example Program

```bash
cargo run --example fft_demo -- demo
cargo run --example fft_demo -- bench 16384 200 impulse
cargo run --example fft_demo -- matrix
```
