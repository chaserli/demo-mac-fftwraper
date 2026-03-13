#![cfg(feature = "parallel")]

mod common;

use common::{generate_input, InputKind};
use std::collections::HashMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use testfftr::{
    fft2_with_workers, fft3_with_workers, fft_batch_with_workers, ifft2_with_workers,
    ifft3_with_workers, ifft_batch_with_workers, AccelerateFFT,
};

#[derive(Debug, Clone, Copy)]
enum BenchOp {
    Forward1d,
    Inverse1d,
    Forward2d,
    Inverse2d,
    Forward3d,
    Inverse3d,
}

impl BenchOp {
    fn as_str(self) -> &'static str {
        match self {
            BenchOp::Forward1d => "forward1d",
            BenchOp::Inverse1d => "inverse1d",
            BenchOp::Forward2d => "forward2d",
            BenchOp::Inverse2d => "inverse2d",
            BenchOp::Forward3d => "forward3d",
            BenchOp::Inverse3d => "inverse3d",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FocusCase {
    label: &'static str,
    input_kind: InputKind,
    iterations: usize,
    width: usize,
    height: usize,
    depth: usize,
    batch: usize,
    workers: i32,
}

impl FocusCase {
    fn total_len(self) -> usize {
        self.width * self.height * self.depth * self.batch
    }
}

#[derive(Debug)]
struct ExternalBenchmark {
    us_per_fft: f64,
}

fn parse_external_benchmark(raw: &str) -> ExternalBenchmark {
    let mut kv = HashMap::<String, String>::new();
    for line in raw.lines() {
        if let Some((k, v)) = line.split_once('=') {
            kv.insert(k.trim().to_string(), v.trim().to_string());
        }
    }

    ExternalBenchmark {
        us_per_fft: kv
            .remove("us_per_fft")
            .expect("missing `us_per_fft` in benchmark output")
            .parse::<f64>()
            .expect("invalid `us_per_fft` value in benchmark output"),
    }
}

fn run_scipy_benchmark(case: FocusCase, op: BenchOp, seed: u64) -> ExternalBenchmark {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = manifest_dir.join("benchmarks").join("scipy_fft_bench.py");

    let mut candidates = Vec::new();
    if let Ok(bin) = std::env::var("PYTHON_BIN") {
        candidates.push(bin);
    } else {
        candidates.push("python3".to_string());
        candidates.push("python".to_string());
    }

    let mut errors = Vec::new();
    for python_bin in candidates {
        let mut cmd = Command::new(&python_bin);
        cmd.arg(script.as_os_str())
            .arg("--op")
            .arg(op.as_str())
            .arg("--iterations")
            .arg(case.iterations.to_string())
            .arg("--samples")
            .arg("9")
            .arg("--workers")
            .arg(case.workers.to_string())
            .arg("--input-kind")
            .arg(case.input_kind.as_str())
            .arg("--seed")
            .arg(seed.to_string());

        match op {
            BenchOp::Forward1d | BenchOp::Inverse1d => {
                cmd.arg("--n")
                    .arg(case.width.to_string())
                    .arg("--batch")
                    .arg(case.batch.to_string());
            }
            BenchOp::Forward2d | BenchOp::Inverse2d => {
                cmd.arg("--width")
                    .arg(case.width.to_string())
                    .arg("--height")
                    .arg(case.height.to_string());
            }
            BenchOp::Forward3d | BenchOp::Inverse3d => {
                cmd.arg("--width")
                    .arg(case.width.to_string())
                    .arg("--height")
                    .arg(case.height.to_string())
                    .arg("--depth")
                    .arg(case.depth.to_string());
            }
        }

        match cmd.output() {
            Ok(output) if output.status.success() => {
                let stdout =
                    String::from_utf8(output.stdout).expect("benchmark output is not valid UTF-8");
                return parse_external_benchmark(&stdout);
            }
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                errors.push(format!(
                    "interpreter `{}` exited with {}.\nstdout:\n{}\nstderr:\n{}",
                    python_bin, output.status, stdout, stderr
                ));
            }
            Err(err) => {
                errors.push(format!(
                    "failed to launch interpreter `{}`: {}",
                    python_bin, err
                ));
            }
        }
    }

    panic!(
        "unable to run scipy benchmark script. attempts:\n{}",
        errors.join("\n\n")
    );
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("benchmark sample was NaN"));
    values[values.len() / 2]
}

fn benchmark_iterations_with_reset<F>(
    baseline: &[num_complex::Complex<f64>],
    iterations: usize,
    samples: usize,
    mut f: F,
) -> f64
where
    F: FnMut(&mut [num_complex::Complex<f64>]),
{
    assert!(iterations > 0, "iterations must be positive");
    assert!(samples > 0, "samples must be positive");

    let mut work = baseline.to_vec();

    for _ in 0..3 {
        work.copy_from_slice(baseline);
        f(&mut work);
        black_box(work[0]);
    }

    let mut us_per_fft_samples = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        for _ in 0..iterations {
            work.copy_from_slice(baseline);
            f(&mut work);
            black_box(work[0]);
        }
        us_per_fft_samples.push(start.elapsed().as_secs_f64() * 1e6 / iterations as f64);
    }

    median(us_per_fft_samples)
}

fn run_accelerate_benchmark(case: FocusCase, op: BenchOp, seed: u64) -> f64 {
    const SAMPLES: usize = 9;

    let baseline = generate_input(case.total_len(), case.input_kind, seed);
    match op {
        BenchOp::Forward1d => benchmark_iterations_with_reset(
            &baseline,
            case.iterations,
            SAMPLES,
            |data| fft_batch_with_workers(data, case.width, case.batch, case.workers),
        ),
        BenchOp::Inverse1d => benchmark_iterations_with_reset(
            &baseline,
            case.iterations,
            SAMPLES,
            |data| ifft_batch_with_workers(data, case.width, case.batch, case.workers),
        ),
        BenchOp::Forward2d => benchmark_iterations_with_reset(
            &baseline,
            case.iterations,
            SAMPLES,
            |data| fft2_with_workers(data, case.width, case.height, case.workers),
        ),
        BenchOp::Inverse2d => benchmark_iterations_with_reset(
            &baseline,
            case.iterations,
            SAMPLES,
            |data| ifft2_with_workers(data, case.width, case.height, case.workers),
        ),
        BenchOp::Forward3d => benchmark_iterations_with_reset(
            &baseline,
            case.iterations,
            SAMPLES,
            |data| fft3_with_workers(data, case.width, case.height, case.depth, case.workers),
        ),
        BenchOp::Inverse3d => benchmark_iterations_with_reset(
            &baseline,
            case.iterations,
            SAMPLES,
            |data| ifft3_with_workers(data, case.width, case.height, case.depth, case.workers),
        ),
    }
}

fn ratio(scipy: f64, accel: f64) -> f64 {
    scipy / accel
}

#[test]
#[ignore = "Focused benchmark table for README updates"]
fn benchmark_focus_cases() {
    assert!(
        !cfg!(debug_assertions),
        "focused benchmark must run in release mode: \
         cargo test --release --features parallel --test benchmark_focus -- --ignored --nocapture"
    );

    let seed = 0x1234_5678_u64;
    let cases = [
        FocusCase {
            label: "1D regular",
            input_kind: InputKind::MultiTone,
            iterations: 6,
            width: 1_048_576,
            height: 1,
            depth: 1,
            batch: 2,
            workers: 2,
        },
        FocusCase {
            label: "1D regular full-core",
            input_kind: InputKind::MultiTone,
            iterations: 6,
            width: 1_048_576,
            height: 1,
            depth: 1,
            batch: 2,
            workers: -1,
        },
        FocusCase {
            label: "2D regular",
            input_kind: InputKind::MultiTone,
            iterations: 4,
            width: 1_024,
            height: 1_024,
            depth: 1,
            batch: 1,
            workers: 2,
        },
        FocusCase {
            label: "2D regular full-core",
            input_kind: InputKind::MultiTone,
            iterations: 4,
            width: 1_024,
            height: 1_024,
            depth: 1,
            batch: 1,
            workers: -1,
        },
        FocusCase {
            label: "2D challenging",
            input_kind: InputKind::Random,
            iterations: 8,
            width: 768,
            height: 512,
            depth: 1,
            batch: 1,
            workers: 2,
        },
        FocusCase {
            label: "2D challenging full-core",
            input_kind: InputKind::Random,
            iterations: 8,
            width: 768,
            height: 512,
            depth: 1,
            batch: 1,
            workers: -1,
        },
        FocusCase {
            label: "3D regular",
            input_kind: InputKind::MultiTone,
            iterations: 3,
            width: 128,
            height: 128,
            depth: 64,
            batch: 1,
            workers: 2,
        },
        FocusCase {
            label: "3D regular full-core",
            input_kind: InputKind::MultiTone,
            iterations: 3,
            width: 128,
            height: 128,
            depth: 64,
            batch: 1,
            workers: -1,
        },
        FocusCase {
            label: "3D challenging",
            input_kind: InputKind::Random,
            iterations: 4,
            width: 96,
            height: 64,
            depth: 32,
            batch: 1,
            workers: 2,
        },
        FocusCase {
            label: "3D challenging full-core",
            input_kind: InputKind::Random,
            iterations: 4,
            width: 96,
            height: 64,
            depth: 32,
            batch: 1,
            workers: -1,
        },
    ];

    println!("| Workload | Input | Workers | Accel Fwd us/op | SciPy Fwd us/op | Fwd Ratio | Accel Inv us/op | SciPy Inv us/op | Inv Ratio |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|");

    for case in cases {
        match (case.depth, case.height) {
            (1, 1) => {
                assert!(AccelerateFFT::try_new(case.width).is_ok());
            }
            (1, _) => {
                assert!(AccelerateFFT::try_new(case.width).is_ok());
                assert!(AccelerateFFT::try_new(case.height).is_ok());
            }
            _ => {
                assert!(AccelerateFFT::try_new(case.width).is_ok());
                assert!(AccelerateFFT::try_new(case.height).is_ok());
                assert!(AccelerateFFT::try_new(case.depth).is_ok());
            }
        }

        let (forward_op, inverse_op) = match (case.depth, case.height) {
            (1, 1) => (BenchOp::Forward1d, BenchOp::Inverse1d),
            (1, _) => (BenchOp::Forward2d, BenchOp::Inverse2d),
            _ => (BenchOp::Forward3d, BenchOp::Inverse3d),
        };

        let accel_fwd = run_accelerate_benchmark(case, forward_op, seed);
        let scipy_fwd = run_scipy_benchmark(case, forward_op, seed).us_per_fft;
        let accel_inv = run_accelerate_benchmark(case, inverse_op, seed);
        let scipy_inv = run_scipy_benchmark(case, inverse_op, seed).us_per_fft;

        println!(
            "| {} | `{}` | `{}` | {:.3} | {:.3} | {:.3}x | {:.3} | {:.3} | {:.3}x |",
            case.label,
            case.input_kind.as_str(),
            case.workers,
            accel_fwd,
            scipy_fwd,
            ratio(scipy_fwd, accel_fwd),
            accel_inv,
            scipy_inv,
            ratio(scipy_inv, accel_inv),
        );
    }
}
