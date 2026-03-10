use num_complex::Complex;
use std::env;
use std::f64::consts::PI;
use std::time::Instant;
use testfftr::{AccelerateFFT, AccelerateFftError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputKind {
    Impulse,
    Ramp,
    MultiTone,
    Chirp,
    Random,
}

impl InputKind {
    const ALL: [InputKind; 5] = [
        InputKind::Impulse,
        InputKind::Ramp,
        InputKind::MultiTone,
        InputKind::Chirp,
        InputKind::Random,
    ];

    fn as_str(self) -> &'static str {
        match self {
            InputKind::Impulse => "impulse",
            InputKind::Ramp => "ramp",
            InputKind::MultiTone => "multitone",
            InputKind::Chirp => "chirp",
            InputKind::Random => "random",
        }
    }

    fn parse(text: &str) -> Option<InputKind> {
        match text {
            "impulse" => Some(InputKind::Impulse),
            "ramp" => Some(InputKind::Ramp),
            "multitone" => Some(InputKind::MultiTone),
            "chirp" => Some(InputKind::Chirp),
            "random" => Some(InputKind::Random),
            _ => None,
        }
    }
}

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn lcg_unit_f64(state: &mut u64) -> f64 {
    let bits = lcg_next(state) >> 11;
    bits as f64 * (1.0 / ((1u64 << 53) as f64))
}

fn generate_input(n: usize, kind: InputKind, seed: u64) -> Vec<Complex<f64>> {
    match kind {
        InputKind::Impulse => {
            let mut out = vec![Complex::new(0.0, 0.0); n];
            out[0] = Complex::new(1.0, 0.0);
            out
        }
        InputKind::Ramp => (0..n)
            .map(|i| {
                let x = i as f64 / n as f64;
                Complex::new(x, 1.0 - x)
            })
            .collect(),
        InputKind::MultiTone => (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let re = (2.0 * PI * 3.0 * t).sin() + 0.5 * (2.0 * PI * 11.0 * t + 0.3).sin();
                let im = (2.0 * PI * 5.0 * t).cos() - 0.25 * (2.0 * PI * 9.0 * t).sin();
                Complex::new(re, im)
            })
            .collect(),
        InputKind::Chirp => (0..n)
            .map(|i| {
                let x = i as f64;
                let phase = PI * x * x / n as f64;
                Complex::new(phase.cos(), phase.sin())
            })
            .collect(),
        InputKind::Random => {
            let mut state = seed;
            (0..n)
                .map(|_| {
                    let re = 2.0 * lcg_unit_f64(&mut state) - 1.0;
                    let im = 2.0 * lcg_unit_f64(&mut state) - 1.0;
                    Complex::new(re, im)
                })
                .collect()
        }
    }
}

fn benchmark_forward(
    plan: &mut AccelerateFFT,
    data: &mut [Complex<f64>],
    iterations: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        plan.forward(data);
    }
    start.elapsed().as_secs_f64() * 1e6 / iterations as f64
}

fn print_usage(program: &str) {
    eprintln!("Usage:");
    eprintln!("  {program} demo");
    eprintln!("  {program} bench <n> <iterations> <input_kind>");
    eprintln!("  {program} matrix");
    eprintln!("Input kinds: impulse, ramp, multitone, chirp, random");
}

fn run_demo() {
    let n = 1024usize;
    let mut fft = AccelerateFFT::new(n);
    let mut data = generate_input(n, InputKind::Ramp, 0x1234_5678);

    fft.forward(&mut data);
    fft.inverse(&mut data);

    println!(
        "Round-trip first sample (unnormalized): {:.6} + {:.6}i",
        data[0].re, data[0].im
    );
}

fn run_bench(n: usize, iterations: usize, kind: InputKind) {
    let mut fft = match AccelerateFFT::try_new(n) {
        Ok(plan) => plan,
        Err(AccelerateFftError::ForwardSetupFailed { .. })
        | Err(AccelerateFftError::InverseSetupFailed { .. }) => {
            eprintln!("Accelerate does not provide a DFT setup for n={n} on this system.");
            return;
        }
        Err(AccelerateFftError::InvalidLength) => {
            eprintln!("n must be greater than zero.");
            return;
        }
    };

    let mut data = generate_input(n, kind, 0x1234_5678);
    let us_per_fft = benchmark_forward(&mut fft, &mut data, iterations);
    println!(
        "accelerate n={} input={} iterations={} us/fft={:.3}",
        n,
        kind.as_str(),
        iterations,
        us_per_fft
    );
}

fn run_matrix() {
    let lengths = [1000usize, 1024, 1536, 4096];
    let iterations = 3000usize;

    for n in lengths {
        let mut fft = match AccelerateFFT::try_new(n) {
            Ok(plan) => plan,
            Err(AccelerateFftError::ForwardSetupFailed { .. })
            | Err(AccelerateFftError::InverseSetupFailed { .. }) => {
                eprintln!("Accelerate does not provide a DFT setup for n={n} on this system.");
                continue;
            }
            Err(AccelerateFftError::InvalidLength) => continue,
        };

        for kind in InputKind::ALL {
            let mut data = generate_input(n, kind, 0x1234_5678);
            let us_per_fft = benchmark_forward(&mut fft, &mut data, iterations);
            println!(
                "accelerate n={} input={} iterations={} us/fft={:.3}",
                n,
                kind.as_str(),
                iterations,
                us_per_fft
            );
        }
    }
}

fn parse_usize(s: &str, name: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("invalid {name}: `{s}`"))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program = args.first().map_or("fft_demo", |s| s.as_str());

    if args.len() < 2 {
        print_usage(program);
        return;
    }

    match args[1].as_str() {
        "demo" => run_demo(),
        "bench" => {
            if args.len() != 5 {
                print_usage(program);
                return;
            }

            let n = match parse_usize(&args[2], "n") {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    return;
                }
            };
            let iterations = match parse_usize(&args[3], "iterations") {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    return;
                }
            };
            let kind = match InputKind::parse(&args[4]) {
                Some(k) => k,
                None => {
                    eprintln!("invalid input_kind: `{}`", args[4]);
                    print_usage(program);
                    return;
                }
            };

            run_bench(n, iterations, kind);
        }
        "matrix" => run_matrix(),
        _ => print_usage(program),
    }
}
