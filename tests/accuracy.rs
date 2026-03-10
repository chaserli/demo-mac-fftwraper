mod common;

use common::{generate_input, max_abs_error, naive_dft_forward, InputKind};
use testfftr::{AccelerateFFT, AccelerateFftError};

#[test]
fn random_input_generation_is_reproducible() {
    let a = generate_input(32, InputKind::Random, 12345);
    let b = generate_input(32, InputKind::Random, 12345);
    let c = generate_input(32, InputKind::Random, 54321);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn try_new_handles_non_radix_lengths_without_panic() {
    let non_radix_lengths = [3usize, 5, 6, 7, 10, 12, 15];
    for n in non_radix_lengths {
        match AccelerateFFT::try_new(n) {
            Ok(_) => {}
            Err(AccelerateFftError::ForwardSetupFailed { n: got }) => assert_eq!(got, n),
            Err(AccelerateFftError::InverseSetupFailed { n: got }) => assert_eq!(got, n),
            Err(AccelerateFftError::InvalidLength) => panic!("n={n} should be valid"),
        }
    }
}

#[test]
fn forward_matches_naive_dft_for_supported_small_lengths() {
    let lengths = [2usize, 3, 4, 5, 7, 8, 9, 16];
    let patterns = InputKind::ALL;

    let mut compared_cases = 0usize;
    for n in lengths {
        let mut fft = match AccelerateFFT::try_new(n) {
            Ok(plan) => plan,
            Err(_) => continue,
        };
        for kind in patterns {
            let input = generate_input(n, kind, 0xDEADBEEF);
            let mut got = input.clone();
            let expected = naive_dft_forward(&input);
            fft.forward(&mut got);

            let err = max_abs_error(&got, &expected);
            let tol = 1e-9 * n as f64;
            assert!(
                err <= tol,
                "n={n}, input={}, max_abs_error={err}, tol={tol}",
                kind.as_str()
            );
            compared_cases += 1;
        }
    }

    assert!(
        compared_cases > 0,
        "No supported lengths were available for naive-DFT comparison."
    );
}

#[test]
fn round_trip_accuracy_on_diverse_inputs() {
    let lengths = [32usize, 128, 1024];
    let patterns = InputKind::ALL;

    for n in lengths {
        let mut fft = AccelerateFFT::new(n);
        for kind in patterns {
            let mut data = generate_input(n, kind, 0x1234_5678);
            let original = data.clone();
            fft.forward(&mut data);
            fft.inverse(&mut data);

            let scale = n as f64;
            let expected: Vec<_> = original.iter().map(|x| *x * scale).collect();
            let err = max_abs_error(&data, &expected);
            let tol = 1e-8 * scale;
            assert!(
                err <= tol,
                "round-trip error too high: n={n}, input={}, err={err}, tol={tol}",
                kind.as_str()
            );
        }
    }
}
