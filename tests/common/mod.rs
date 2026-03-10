#![allow(dead_code)]

use num_complex::Complex;
use std::f64::consts::PI;
use std::time::{Duration, Instant};
use testfftr::AccelerateFFT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Impulse,
    Ramp,
    MultiTone,
    Chirp,
    Random,
}

impl InputKind {
    pub const ALL: [InputKind; 5] = [
        InputKind::Impulse,
        InputKind::Ramp,
        InputKind::MultiTone,
        InputKind::Chirp,
        InputKind::Random,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            InputKind::Impulse => "impulse",
            InputKind::Ramp => "ramp",
            InputKind::MultiTone => "multitone",
            InputKind::Chirp => "chirp",
            InputKind::Random => "random",
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

pub fn generate_input(n: usize, kind: InputKind, seed: u64) -> Vec<Complex<f64>> {
    assert!(n > 0, "n must be greater than zero");
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

pub fn naive_dft_forward(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    let mut out = vec![Complex::new(0.0, 0.0); n];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut sum = Complex::new(0.0, 0.0);
        for (t, &x) in input.iter().enumerate() {
            let theta = -2.0 * PI * (k as f64) * (t as f64) / (n as f64);
            let w = Complex::new(theta.cos(), theta.sin());
            sum += x * w;
        }
        *slot = sum;
    }
    out
}

pub fn max_abs_error(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0_f64, f64::max)
}

pub fn benchmark_forward(
    plan: &mut AccelerateFFT,
    data: &mut [Complex<f64>],
    iterations: usize,
) -> (Duration, f64) {
    assert!(iterations > 0, "iterations must be greater than zero");
    assert_eq!(
        data.len(),
        plan.len(),
        "benchmark data length mismatch: expected {}, got {}",
        plan.len(),
        data.len()
    );

    let start = Instant::now();
    for _ in 0..iterations {
        plan.forward(data);
    }
    let elapsed = start.elapsed();
    let us_per_fft = elapsed.as_secs_f64() * 1e6 / iterations as f64;
    (elapsed, us_per_fft)
}
