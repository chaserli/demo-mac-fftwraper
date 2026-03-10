#![cfg(feature = "parallel")]

mod common;

use common::{generate_input, max_abs_error, InputKind};
use num_complex::Complex;
use testfftr::{
    fft2, fft2_with_workers, fft3, fft3_with_workers, fft_batch_with_workers, ifft2,
    ifft2_with_workers, ifft3, ifft3_with_workers, ifft_batch_with_workers, AccelerateFFT,
};

fn approx_eq(lhs: &[Complex<f64>], rhs: &[Complex<f64>], tol: f64) {
    assert_eq!(lhs.len(), rhs.len());
    let err = max_abs_error(lhs, rhs);
    assert!(err <= tol, "max_abs_error={err}, tol={tol}");
}

fn workers_matrix() -> Vec<i32> {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut workers = vec![1, 2, -1];
    if available >= 2 {
        workers.push(-2);
    }
    workers
}

#[test]
fn fft_batch_with_workers_matches_plan_forward_inverse() {
    let n = 128usize;
    let batch = 16usize;
    let total = n * batch;

    for kind in InputKind::ALL {
        for workers in workers_matrix() {
            let mut forward_parallel = generate_input(total, kind, 0xA1B2_C3D4);
            let mut forward_seq = forward_parallel.clone();

            let mut seq_fft = AccelerateFFT::new(n);
            seq_fft.forward_batch(&mut forward_seq, batch);
            fft_batch_with_workers(&mut forward_parallel, n, batch, workers);
            approx_eq(&forward_parallel, &forward_seq, 1e-9 * n as f64);

            let mut inverse_parallel = generate_input(total, kind, 0xC0FF_EE00);
            let mut inverse_seq = inverse_parallel.clone();

            for chunk in inverse_seq.chunks_exact_mut(n) {
                seq_fft.inverse(chunk);
            }
            ifft_batch_with_workers(&mut inverse_parallel, n, batch, workers);
            approx_eq(&inverse_parallel, &inverse_seq, 1e-9 * n as f64);
        }
    }
}

#[test]
fn fft_batch_with_workers_round_trip_scaled() {
    let n = 128usize;
    let batch = 20usize;
    let total = n * batch;
    let scale = n as f64;

    for kind in InputKind::ALL {
        for workers in workers_matrix() {
            let mut data = generate_input(total, kind, 0x55AA_11EE);
            let original = data.clone();

            fft_batch_with_workers(&mut data, n, batch, workers);
            ifft_batch_with_workers(&mut data, n, batch, workers);

            let expected: Vec<_> = original.iter().map(|v| *v * scale).collect();
            approx_eq(&data, &expected, 1e-8 * scale);
        }
    }
}

#[test]
fn fft2_with_workers_matches_simple_forward_inverse() {
    let width = 16usize;
    let height = 8usize;
    let n = width * height;
    let tol = 1e-8 * n as f64;

    for kind in InputKind::ALL {
        for workers in workers_matrix() {
            let mut forward_parallel = generate_input(n, kind, 0xABCD_0011);
            let mut forward_seq = forward_parallel.clone();
            fft2(&mut forward_seq, width, height);
            fft2_with_workers(&mut forward_parallel, width, height, workers);
            approx_eq(&forward_parallel, &forward_seq, tol);

            let mut inverse_parallel = generate_input(n, kind, 0xABCD_0012);
            let mut inverse_seq = inverse_parallel.clone();
            ifft2(&mut inverse_seq, width, height);
            ifft2_with_workers(&mut inverse_parallel, width, height, workers);
            approx_eq(&inverse_parallel, &inverse_seq, tol);
        }
    }
}

#[test]
fn fft2_with_workers_round_trip_scaled() {
    let width = 16usize;
    let height = 16usize;
    let total = width * height;
    let scale = total as f64;

    for kind in InputKind::ALL {
        for workers in workers_matrix() {
            let mut data = generate_input(total, kind, 0x4455_6677);
            let original = data.clone();
            fft2_with_workers(&mut data, width, height, workers);
            ifft2_with_workers(&mut data, width, height, workers);

            let expected: Vec<_> = original.iter().map(|v| *v * scale).collect();
            approx_eq(&data, &expected, 1e-8 * scale);
        }
    }
}

#[test]
fn fft3_with_workers_matches_simple_forward_inverse() {
    let width = 8usize;
    let height = 8usize;
    let depth = 4usize;
    let total = width * height * depth;
    let tol = 1e-8 * total as f64;

    for kind in InputKind::ALL {
        for workers in workers_matrix() {
            let mut forward_parallel = generate_input(total, kind, 0x1020_3040);
            let mut forward_seq = forward_parallel.clone();
            fft3(&mut forward_seq, width, height, depth);
            fft3_with_workers(&mut forward_parallel, width, height, depth, workers);
            approx_eq(&forward_parallel, &forward_seq, tol);

            let mut inverse_parallel = generate_input(total, kind, 0x1020_3041);
            let mut inverse_seq = inverse_parallel.clone();
            ifft3(&mut inverse_seq, width, height, depth);
            ifft3_with_workers(&mut inverse_parallel, width, height, depth, workers);
            approx_eq(&inverse_parallel, &inverse_seq, tol);
        }
    }
}

#[test]
fn fft3_with_workers_round_trip_scaled() {
    let width = 8usize;
    let height = 8usize;
    let depth = 8usize;
    let total = width * height * depth;
    let scale = total as f64;

    for kind in InputKind::ALL {
        for workers in workers_matrix() {
            let mut data = generate_input(total, kind, 0xBEEF_F00D);
            let original = data.clone();
            fft3_with_workers(&mut data, width, height, depth, workers);
            ifft3_with_workers(&mut data, width, height, depth, workers);

            let expected: Vec<_> = original.iter().map(|v| *v * scale).collect();
            approx_eq(&data, &expected, 1e-8 * scale);
        }
    }
}

#[test]
fn fft3_with_workers_matches_simple_api() {
    let width = 16usize;
    let height = 8usize;
    let depth = 4usize;
    let total = width * height * depth;
    let tol = 1e-8 * total as f64;

    for kind in InputKind::ALL {
        for workers in workers_matrix() {
            let mut parallel = generate_input(total, kind, 0x5151_6161);
            let mut sequential = parallel.clone();

            fft3(&mut sequential, width, height, depth);
            fft3_with_workers(&mut parallel, width, height, depth, workers);
            approx_eq(&parallel, &sequential, tol);

            ifft3(&mut sequential, width, height, depth);
            ifft3_with_workers(&mut parallel, width, height, depth, workers);
            let scale = total as f64;
            let expected: Vec<_> = generate_input(total, kind, 0x5151_6161)
                .iter()
                .map(|v| *v * scale)
                .collect();
            approx_eq(&sequential, &expected, 1e-8 * scale);
            approx_eq(&parallel, &expected, 1e-8 * scale);
        }
    }
}

#[test]
#[should_panic(expected = "workers must not be zero")]
fn workers_zero_panics() {
    let mut data = generate_input(64, InputKind::Random, 0x1234);
    fft2_with_workers(&mut data, 8, 8, 0);
}

#[test]
fn workers_negative_one_and_negative_two_valid_when_supported() {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut data = generate_input(64, InputKind::Random, 0x5678);
    fft2_with_workers(&mut data, 8, 8, -1);

    if available >= 2 {
        ifft2_with_workers(&mut data, 8, 8, -2);
    }
}

#[test]
#[should_panic(expected = "effective worker count must be at least 1")]
fn workers_too_negative_panics() {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1) as i32;
    let workers = -(available + 2);
    let mut data = generate_input(64, InputKind::Random, 0x9999);
    fft2_with_workers(&mut data, 8, 8, workers);
}

#[test]
#[should_panic(expected = "expects data length")]
fn fft_batch_with_workers_panics_on_shape_mismatch() {
    let mut data = generate_input(63, InputKind::Ramp, 0xDEAD);
    fft_batch_with_workers(&mut data, 8, 8, 1);
}

#[test]
#[should_panic(expected = "fft2_with_workers expects data length")]
fn fft2_with_workers_panics_on_shape_mismatch() {
    let mut data = generate_input(63, InputKind::Ramp, 0xBEEF);
    fft2_with_workers(&mut data, 8, 8, 1);
}

#[test]
#[should_panic(expected = "fft3_with_workers expects data length")]
fn fft3_with_workers_panics_on_shape_mismatch() {
    let mut data = generate_input(100, InputKind::Ramp, 0xCAFE);
    fft3_with_workers(&mut data, 5, 5, 5, 1);
}
