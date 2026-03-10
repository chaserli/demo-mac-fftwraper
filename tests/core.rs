mod common;

use common::{benchmark_forward, generate_input, InputKind};
use num_complex::Complex;
use testfftr::{fft, fft2, fft3, ifft, ifft2, ifft3, AccelerateFFT};

fn assert_close(lhs: Complex<f64>, rhs: Complex<f64>, tol: f64) {
    let diff = (lhs - rhs).norm();
    assert!(
        diff <= tol,
        "complex mismatch: lhs={lhs:?}, rhs={rhs:?}, diff={diff}, tol={tol}"
    );
}

#[test]
#[should_panic(expected = "FFT length must be greater than zero")]
fn new_panics_on_zero_length() {
    let _ = AccelerateFFT::new(0);
}

#[test]
#[should_panic(expected = "FFT plan length mismatch")]
fn forward_panics_on_length_mismatch() {
    let mut fft = AccelerateFFT::new(8);
    let mut data = vec![Complex::new(0.0, 0.0); 4];
    fft.forward(&mut data);
}

#[test]
fn forward_batch_matches_sequential_forward() {
    let n = 16usize;
    let batch = 4usize;
    let mut data = generate_input(n * batch, InputKind::MultiTone, 0xAA55);
    let mut expected = data.clone();

    let mut batch_plan = AccelerateFFT::new(n);
    batch_plan.forward_batch(&mut data, batch);

    let mut seq_plan = AccelerateFFT::new(n);
    for chunk in expected.chunks_exact_mut(n) {
        seq_plan.forward(chunk);
    }

    for (got, want) in data.iter().zip(expected.iter()) {
        assert_close(*got, *want, 1e-9);
    }
}

#[test]
fn fft2_matches_explicit_row_then_column() {
    let width = 8usize;
    let height = 4usize;
    let mut data = generate_input(width * height, InputKind::Random, 0x1234_5678);
    let mut expected = data.clone();

    let mut row_fft = AccelerateFFT::new(width);
    for row in expected.chunks_exact_mut(width) {
        row_fft.forward(row);
    }

    let mut col_fft = AccelerateFFT::new(height);
    let mut column = vec![Complex::new(0.0, 0.0); height];
    for x in 0..width {
        for y in 0..height {
            column[y] = expected[y * width + x];
        }
        col_fft.forward(&mut column);
        for y in 0..height {
            expected[y * width + x] = column[y];
        }
    }

    fft2(&mut data, width, height);

    for (got, want) in data.iter().zip(expected.iter()) {
        assert_close(*got, *want, 1e-9);
    }
}

#[test]
fn benchmark_helper_reports_positive_throughput() {
    let n = 64usize;
    let mut fft = AccelerateFFT::new(n);
    let mut data = generate_input(n, InputKind::Ramp, 0x1234);
    let (_elapsed, us_per_fft) = benchmark_forward(&mut fft, &mut data, 64);
    assert!(us_per_fft.is_finite());
    assert!(us_per_fft > 0.0);
}

#[test]
fn fft1d_helpers_round_trip_scaled() {
    let n = 64usize;
    let mut data = generate_input(n, InputKind::Chirp, 0xCAFE_BABE);
    let original = data.clone();

    fft(&mut data);
    ifft(&mut data);

    let scale = n as f64;
    for (got, orig) in data.iter().zip(original.iter()) {
        assert_close(*got, *orig * scale, 1e-8 * scale);
    }
}

#[test]
fn fft2d_forward_inverse_round_trip_scaled() {
    let width = 16usize;
    let height = 8usize;
    let mut data = generate_input(width * height, InputKind::Random, 0xA5A5_1234);
    let original = data.clone();

    fft2(&mut data, width, height);
    ifft2(&mut data, width, height);

    let scale = (width * height) as f64;
    for (got, orig) in data.iter().zip(original.iter()) {
        assert_close(*got, *orig * scale, 1e-8 * scale);
    }
}

#[test]
fn fft3_round_trip_scaled() {
    let width = 8usize;
    let height = 4usize;
    let depth = 4usize;
    let mut data = generate_input(width * height * depth, InputKind::MultiTone, 0xFACE_B00C);
    let original = data.clone();

    fft3(&mut data, width, height, depth);
    ifft3(&mut data, width, height, depth);

    let scale = (width * height * depth) as f64;
    for (got, orig) in data.iter().zip(original.iter()) {
        assert_close(*got, *orig * scale, 1e-8 * scale);
    }
}

#[test]
fn fft3_matches_plan_equivalent() {
    let mut data_3d = generate_input(8 * 4 * 2, InputKind::MultiTone, 0x3030);
    let mut expect_3d = data_3d.clone();
    let mut x_fft = AccelerateFFT::new(8);
    for z in 0..2usize {
        let z_base = z * (8 * 4);
        for y in 0..4usize {
            let start = z_base + y * 8;
            let end = start + 8;
            x_fft.forward(&mut expect_3d[start..end]);
        }
    }
    let mut y_fft = AccelerateFFT::new(4);
    let mut y_line = vec![Complex::new(0.0, 0.0); 4];
    for z in 0..2usize {
        let z_base = z * (8 * 4);
        for x in 0..8usize {
            for y in 0..4usize {
                y_line[y] = expect_3d[z_base + y * 8 + x];
            }
            y_fft.forward(&mut y_line);
            for y in 0..4usize {
                expect_3d[z_base + y * 8 + x] = y_line[y];
            }
        }
    }
    let mut z_fft = AccelerateFFT::new(2);
    let mut z_line = vec![Complex::new(0.0, 0.0); 2];
    for y in 0..4usize {
        for x in 0..8usize {
            for z in 0..2usize {
                z_line[z] = expect_3d[z * (8 * 4) + y * 8 + x];
            }
            z_fft.forward(&mut z_line);
            for z in 0..2usize {
                expect_3d[z * (8 * 4) + y * 8 + x] = z_line[z];
            }
        }
    }
    fft3(&mut data_3d, 8, 4, 2);
    for (got, want) in data_3d.iter().zip(expect_3d.iter()) {
        assert_close(*got, *want, 1e-9);
    }
}

#[test]
#[should_panic(expected = "fft3 expects data length")]
fn fft3_panics_on_shape_mismatch() {
    let mut data = generate_input(63, InputKind::Random, 0x1234);
    fft3(&mut data, 4, 4, 4);
}
