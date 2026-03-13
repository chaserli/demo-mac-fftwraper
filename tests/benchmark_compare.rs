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
struct Case {
    op: BenchOp,
    input_kind: InputKind,
    iterations: usize,
    width: usize,
    height: usize,
    depth: usize,
    batch: usize,
    workers: i32,
}

impl Case {
    fn total_len(self) -> usize {
        self.width * self.height * self.depth * self.batch
    }

    fn independent_jobs(self) -> usize {
        match self.op {
            BenchOp::Forward1d | BenchOp::Inverse1d => self.batch,
            BenchOp::Forward2d | BenchOp::Inverse2d => self.height,
            BenchOp::Forward3d | BenchOp::Inverse3d => self.height * self.depth,
        }
    }

    fn dims_label(self) -> String {
        match self.op {
            BenchOp::Forward1d | BenchOp::Inverse1d => {
                format!("n={} batch={}", self.width, self.batch)
            }
            BenchOp::Forward2d | BenchOp::Inverse2d => {
                format!("w={} h={}", self.width, self.height)
            }
            BenchOp::Forward3d | BenchOp::Inverse3d => {
                format!("w={} h={} d={}", self.width, self.height, self.depth)
            }
        }
    }
}

fn effective_workers(workers: i32, jobs: usize) -> usize {
    assert!(workers != 0);
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1) as i32;
    let resolved = if workers > 0 {
        workers
    } else if workers == -1 {
        available
    } else {
        available - ((-workers) - 1)
    };
    assert!(resolved >= 1);
    (resolved as usize).min(jobs)
}

#[derive(Debug)]
struct ExternalBenchmark {
    backend: String,
    op: String,
    iterations: usize,
    input_kind: String,
    workers: i32,
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
        backend: kv
            .remove("backend")
            .expect("missing `backend` in benchmark output"),
        op: kv.remove("op").expect("missing `op` in benchmark output"),
        iterations: kv
            .remove("iterations")
            .expect("missing `iterations` in benchmark output")
            .parse::<usize>()
            .expect("invalid `iterations` value in benchmark output"),
        input_kind: kv
            .remove("input_kind")
            .expect("missing `input_kind` in benchmark output"),
        workers: kv
            .remove("workers")
            .expect("missing `workers` in benchmark output")
            .parse::<i32>()
            .expect("invalid `workers` value in benchmark output"),
        us_per_fft: kv
            .remove("us_per_fft")
            .expect("missing `us_per_fft` in benchmark output")
            .parse::<f64>()
            .expect("invalid `us_per_fft` value in benchmark output"),
    }
}

fn run_scipy_benchmark(case: Case, seed: u64) -> ExternalBenchmark {
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
            .arg(case.op.as_str())
            .arg("--iterations")
            .arg(case.iterations.to_string())
            .arg("--samples")
            .arg("5")
            .arg("--workers")
            .arg(case.workers.to_string())
            .arg("--input-kind")
            .arg(case.input_kind.as_str())
            .arg("--seed")
            .arg(seed.to_string());

        match case.op {
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

fn run_accelerate_benchmark(case: Case, seed: u64) -> Option<f64> {
    const SAMPLES: usize = 5;

    match case.op {
        BenchOp::Forward1d => {
            if AccelerateFFT::try_new(case.width).is_err() {
                return None;
            }
            let baseline = generate_input(case.total_len(), case.input_kind, seed);
            Some(benchmark_iterations_with_reset(
                &baseline,
                case.iterations,
                SAMPLES,
                |data| fft_batch_with_workers(data, case.width, case.batch, case.workers),
            ))
        }
        BenchOp::Inverse1d => {
            if AccelerateFFT::try_new(case.width).is_err() {
                return None;
            }
            let baseline = generate_input(case.total_len(), case.input_kind, seed);
            Some(benchmark_iterations_with_reset(
                &baseline,
                case.iterations,
                SAMPLES,
                |data| ifft_batch_with_workers(data, case.width, case.batch, case.workers),
            ))
        }
        BenchOp::Forward2d => {
            if AccelerateFFT::try_new(case.width).is_err()
                || AccelerateFFT::try_new(case.height).is_err()
            {
                return None;
            }
            let baseline = generate_input(case.total_len(), case.input_kind, seed);
            Some(benchmark_iterations_with_reset(
                &baseline,
                case.iterations,
                SAMPLES,
                |data| fft2_with_workers(data, case.width, case.height, case.workers),
            ))
        }
        BenchOp::Inverse2d => {
            if AccelerateFFT::try_new(case.width).is_err()
                || AccelerateFFT::try_new(case.height).is_err()
            {
                return None;
            }
            let baseline = generate_input(case.total_len(), case.input_kind, seed);
            Some(benchmark_iterations_with_reset(
                &baseline,
                case.iterations,
                SAMPLES,
                |data| ifft2_with_workers(data, case.width, case.height, case.workers),
            ))
        }
        BenchOp::Forward3d => {
            if AccelerateFFT::try_new(case.width).is_err()
                || AccelerateFFT::try_new(case.height).is_err()
                || AccelerateFFT::try_new(case.depth).is_err()
            {
                return None;
            }
            let baseline = generate_input(case.total_len(), case.input_kind, seed);
            Some(benchmark_iterations_with_reset(
                &baseline,
                case.iterations,
                SAMPLES,
                |data| fft3_with_workers(data, case.width, case.height, case.depth, case.workers),
            ))
        }
        BenchOp::Inverse3d => {
            if AccelerateFFT::try_new(case.width).is_err()
                || AccelerateFFT::try_new(case.height).is_err()
                || AccelerateFFT::try_new(case.depth).is_err()
            {
                return None;
            }
            let baseline = generate_input(case.total_len(), case.input_kind, seed);
            Some(benchmark_iterations_with_reset(
                &baseline,
                case.iterations,
                SAMPLES,
                |data| ifft3_with_workers(data, case.width, case.height, case.depth, case.workers),
            ))
        }
    }
}

#[test]
#[ignore = "Long-running benchmark comparison; run with --ignored --nocapture"]
fn compare_accelerate_with_scipy_all_ops() {
    assert!(
        !cfg!(debug_assertions),
        "benchmark comparison must be run in release mode: \
         cargo test --release --features parallel --test benchmark_compare -- --ignored --nocapture"
    );

    let inputs = [
        InputKind::Impulse,
        InputKind::Ramp,
        InputKind::MultiTone,
        InputKind::Random,
    ];
    let workers_set = [1i32, 2i32, -1i32];
    let seed = 0x1234_5678_u64;
    let mut cases = Vec::<Case>::new();

    for workers in workers_set {
        for (n, batch, iterations) in [
            (16_384usize, 8usize, 220usize),
            (65_536usize, 4usize, 72usize),
            (1_048_576usize, 2usize, 6usize),
        ] {
            for input in inputs {
                cases.push(Case {
                    op: BenchOp::Forward1d,
                    input_kind: input,
                    iterations,
                    width: n,
                    height: 1,
                    depth: 1,
                    batch,
                    workers,
                });
                cases.push(Case {
                    op: BenchOp::Inverse1d,
                    input_kind: input,
                    iterations,
                    width: n,
                    height: 1,
                    depth: 1,
                    batch,
                    workers,
                });
            }
        }

        for (w, h, iterations) in [(512usize, 512usize, 12usize), (768usize, 512usize, 8usize)] {
            for input in inputs {
                cases.push(Case {
                    op: BenchOp::Forward2d,
                    input_kind: input,
                    iterations,
                    width: w,
                    height: h,
                    depth: 1,
                    batch: 1,
                    workers,
                });
                cases.push(Case {
                    op: BenchOp::Inverse2d,
                    input_kind: input,
                    iterations,
                    width: w,
                    height: h,
                    depth: 1,
                    batch: 1,
                    workers,
                });
            }
        }

        for (w, h, d, iterations) in [
            (64usize, 64usize, 32usize, 6usize),
            (96usize, 64usize, 32usize, 4usize),
        ] {
            for input in inputs {
                cases.push(Case {
                    op: BenchOp::Forward3d,
                    input_kind: input,
                    iterations,
                    width: w,
                    height: h,
                    depth: d,
                    batch: 1,
                    workers,
                });
                cases.push(Case {
                    op: BenchOp::Inverse3d,
                    input_kind: input,
                    iterations,
                    width: w,
                    height: h,
                    depth: d,
                    batch: 1,
                    workers,
                });
            }
        }
    }

    let mut executed_cases = 0usize;
    let mut skipped_cases = 0usize;

    for case in cases {
        let accel_us = match run_accelerate_benchmark(case, seed) {
            Some(v) => v,
            None => {
                println!(
                    "Skipping unsupported case: op={} {}, input={}, workers={}",
                    case.op.as_str(),
                    case.dims_label(),
                    case.input_kind.as_str(),
                    case.workers
                );
                skipped_cases += 1;
                continue;
            }
        };

        let scipy = run_scipy_benchmark(case, seed);
        assert_eq!(scipy.op, case.op.as_str());
        assert_eq!(scipy.iterations, case.iterations);
        assert_eq!(scipy.input_kind, case.input_kind.as_str());
        assert_eq!(scipy.workers, case.workers);
        assert!(accel_us.is_finite() && accel_us > 0.0);
        assert!(scipy.us_per_fft.is_finite() && scipy.us_per_fft > 0.0);

        let ratio = scipy.us_per_fft / accel_us;
        let effective_workers = effective_workers(case.workers, case.independent_jobs());
        println!(
            "op={}, {}, input={}, workers(requested={} effective={}): accelerate={:.3} us/op, {}={:.3} us/op, ratio(scipy/accelerate)={:.3}x",
            case.op.as_str(),
            case.dims_label(),
            case.input_kind.as_str(),
            case.workers,
            effective_workers,
            accel_us,
            scipy.backend,
            scipy.us_per_fft,
            ratio
        );
        executed_cases += 1;
    }

    assert!(executed_cases > 0, "No benchmark cases were executed.");
    if skipped_cases > 0 {
        println!("Skipped {} unsupported benchmark cases.", skipped_cases);
    }
}
