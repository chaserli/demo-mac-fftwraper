#[cfg(not(target_os = "macos"))]
compile_error!("This crate requires macOS and Apple Accelerate.");

use libc::c_void;
use num_complex::Complex;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use rayon::ThreadPoolBuilder;
#[cfg(feature = "parallel")]
use std::cell::RefCell;
#[cfg(feature = "parallel")]
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
#[cfg(feature = "parallel")]
use std::sync::{Arc, Mutex, OnceLock};

#[allow(non_camel_case_types, non_upper_case_globals, dead_code)]
mod ffi {
    use libc::{c_int, c_long};

    #[repr(C)]
    pub struct DSPDoubleComplex {
        pub real: f64,
        pub imag: f64,
    }

    #[repr(C)]
    pub struct DSPDoubleSplitComplex {
        pub realp: *mut f64,
        pub imagp: *mut f64,
    }

    #[repr(C)]
    pub struct vDSP_DFT_SetupStructD {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct OpaqueFFTSetupD {
        _private: [u8; 0],
    }

    pub type vDSP_DFT_SetupD = *mut vDSP_DFT_SetupStructD;
    pub type vDSP_DFT_Direction = c_int;
    pub type FFTSetupD = *mut OpaqueFFTSetupD;
    pub type FFTDirection = c_int;
    pub type FFTRadix = c_int;
    pub type vDSP_Length = usize;
    pub type vDSP_Stride = c_long;

    pub const vDSP_DFT_FORWARD: vDSP_DFT_Direction = 1;
    pub const vDSP_DFT_INVERSE: vDSP_DFT_Direction = -1;
    pub const kFFTDirection_Forward: FFTDirection = 1;
    pub const kFFTDirection_Inverse: FFTDirection = -1;
    pub const kFFTRadix2: FFTRadix = 0;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn vDSP_ctozD(
            c: *const DSPDoubleComplex,
            ic: vDSP_Stride,
            z: *const DSPDoubleSplitComplex,
            iz: vDSP_Stride,
            n: vDSP_Length,
        );
        pub fn vDSP_ztocD(
            z: *const DSPDoubleSplitComplex,
            iz: vDSP_Stride,
            c: *mut DSPDoubleComplex,
            ic: vDSP_Stride,
            n: vDSP_Length,
        );

        pub fn vDSP_create_fftsetupD(log2n: vDSP_Length, radix: FFTRadix) -> FFTSetupD;
        pub fn vDSP_destroy_fftsetupD(setup: FFTSetupD);
        pub fn vDSP_fft_zipD(
            setup: FFTSetupD,
            c: *const DSPDoubleSplitComplex,
            stride: vDSP_Stride,
            log2n: vDSP_Length,
            direction: FFTDirection,
        );

        pub fn vDSP_DFT_zop_CreateSetupD(
            previous: vDSP_DFT_SetupD,
            length: vDSP_Length,
            direction: vDSP_DFT_Direction,
        ) -> vDSP_DFT_SetupD;

        pub fn vDSP_DFT_ExecuteD(
            setup: *const vDSP_DFT_SetupStructD,
            ir: *const f64,
            ii: *const f64,
            or_: *mut f64,
            oi: *mut f64,
        );

        pub fn vDSP_DFT_DestroySetupD(setup: vDSP_DFT_SetupD);
    }
}

/// FFT plan for complex128 data backed by Apple Accelerate vDSP DFT.
///
/// Accelerate's DFT APIs operate on split-complex memory (`real[]` and
/// `imag[]` arrays) rather than interleaved complex values.
/// This is friendlier to SIMD pipelines because real and imaginary streams can
/// be loaded and processed independently with less lane shuffling.
///
/// This wrapper converts between interleaved `Complex<f64>` and split-complex
/// buffers. The conversion is a copy cost per call, but buffers are allocated
/// once in the plan and reused for every execution to avoid repeated heap work.
pub struct AccelerateFFT {
    n: usize,
    backend: PlanBackend,
    real_buffer: Vec<f64>,
    imag_buffer: Vec<f64>,
}

enum PlanBackend {
    Dft {
        setup_forward: *mut c_void,
        setup_inverse: *mut c_void,
    },
    Fft {
        setup: *mut c_void,
        log2n: usize,
    },
    MixedRadix3Pow2 {
        n1: usize,
        setup_pow2: *mut c_void,
        log2n1: usize,
        stage_real: Vec<f64>,
        stage_imag: Vec<f64>,
        twiddle_forward_r1: Vec<Complex<f64>>,
        twiddle_forward_r2: Vec<Complex<f64>>,
        twiddle_inverse_r1: Vec<Complex<f64>>,
        twiddle_inverse_r2: Vec<Complex<f64>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccelerateFftError {
    InvalidLength,
    ForwardSetupFailed { n: usize },
    InverseSetupFailed { n: usize },
}

impl fmt::Display for AccelerateFftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccelerateFftError::InvalidLength => {
                write!(f, "FFT length must be greater than zero")
            }
            AccelerateFftError::ForwardSetupFailed { n } => {
                write!(f, "failed to create forward vDSP DFT setup for length {n}")
            }
            AccelerateFftError::InverseSetupFailed { n } => {
                write!(f, "failed to create inverse vDSP DFT setup for length {n}")
            }
        }
    }
}

impl Error for AccelerateFftError {}

impl AccelerateFFT {
    /// Creates a reusable FFT plan for transforms of length `n`.
    ///
    /// Returns a typed error instead of panicking, useful for probing support
    /// of non-radix lengths on a specific Accelerate version/CPU.
    pub fn try_new(n: usize) -> Result<AccelerateFFT, AccelerateFftError> {
        if n == 0 {
            return Err(AccelerateFftError::InvalidLength);
        }

        let real_buffer = vec![0.0; n];
        let imag_buffer = vec![0.0; n];

        // Prefer radix-2 FFT kernels when possible for better performance.
        if n >= 2 && n.is_power_of_two() {
            let log2n = n.trailing_zeros() as usize;
            let setup_fft = unsafe { ffi::vDSP_create_fftsetupD(log2n, ffi::kFFTRadix2) };
            if !setup_fft.is_null() {
                return Ok(AccelerateFFT {
                    n,
                    backend: PlanBackend::Fft {
                        setup: setup_fft.cast::<c_void>(),
                        log2n,
                    },
                    real_buffer,
                    imag_buffer,
                });
            }
        }

        // Fast mixed-radix path for N = 3 * 2^m (for example 1536 = 3 * 512).
        if n % 3 == 0 {
            let n1 = n / 3;
            if n1 >= 2 && n1.is_power_of_two() {
                let log2n1 = n1.trailing_zeros() as usize;
                let setup_pow2 = unsafe { ffi::vDSP_create_fftsetupD(log2n1, ffi::kFFTRadix2) };
                if !setup_pow2.is_null() {
                    let mut twiddle_forward_r1 = Vec::with_capacity(n1);
                    let mut twiddle_forward_r2 = Vec::with_capacity(n1);
                    let mut twiddle_inverse_r1 = Vec::with_capacity(n1);
                    let mut twiddle_inverse_r2 = Vec::with_capacity(n1);
                    let n_f64 = n as f64;
                    for k in 0..n1 {
                        let angle = -2.0 * std::f64::consts::PI * (k as f64) / n_f64;
                        let w1 = Complex::new(angle.cos(), angle.sin());
                        let w2 = Complex::new((2.0 * angle).cos(), (2.0 * angle).sin());
                        twiddle_forward_r1.push(w1);
                        twiddle_forward_r2.push(w2);
                        twiddle_inverse_r1.push(w1.conj());
                        twiddle_inverse_r2.push(w2.conj());
                    }

                    return Ok(AccelerateFFT {
                        n,
                        backend: PlanBackend::MixedRadix3Pow2 {
                            n1,
                            setup_pow2: setup_pow2.cast::<c_void>(),
                            log2n1,
                            stage_real: vec![0.0; n],
                            stage_imag: vec![0.0; n],
                            twiddle_forward_r1,
                            twiddle_forward_r2,
                            twiddle_inverse_r1,
                            twiddle_inverse_r2,
                        },
                        real_buffer,
                        imag_buffer,
                    });
                }
            }
        }

        let setup_forward = unsafe {
            ffi::vDSP_DFT_zop_CreateSetupD(std::ptr::null_mut(), n, ffi::vDSP_DFT_FORWARD)
        };
        if setup_forward.is_null() {
            return Err(AccelerateFftError::ForwardSetupFailed { n });
        }

        let setup_inverse =
            unsafe { ffi::vDSP_DFT_zop_CreateSetupD(setup_forward, n, ffi::vDSP_DFT_INVERSE) };
        if setup_inverse.is_null() {
            unsafe { ffi::vDSP_DFT_DestroySetupD(setup_forward) };
            return Err(AccelerateFftError::InverseSetupFailed { n });
        }

        Ok(AccelerateFFT {
            n,
            backend: PlanBackend::Dft {
                setup_forward: setup_forward.cast::<c_void>(),
                setup_inverse: setup_inverse.cast::<c_void>(),
            },
            real_buffer,
            imag_buffer,
        })
    }

    /// Creates a reusable FFT plan for transforms of length `n`.
    ///
    /// Panics if `n == 0` or if Accelerate fails to create either setup.
    pub fn new(n: usize) -> AccelerateFFT {
        AccelerateFFT::try_new(n).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Returns the transform length this plan was created for.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns true when this plan length is zero.
    ///
    /// This is always false for valid plans, but provided for API completeness.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Executes a forward complex-to-complex FFT in-place on `data`.
    ///
    /// Panics if `data.len() != n`.
    pub fn forward(&mut self, data: &mut [Complex<f64>]) {
        self.execute(data, ffi::kFFTDirection_Forward);
    }

    /// Executes an inverse complex-to-complex FFT in-place on `data`.
    ///
    /// This method is intentionally unnormalized (FFTW-style). A round-trip
    /// `inverse(forward(x))` yields approximately `n * x`.
    ///
    /// Panics if `data.len() != n`.
    pub fn inverse(&mut self, data: &mut [Complex<f64>]) {
        self.execute(data, ffi::kFFTDirection_Inverse);
    }

    /// Executes `batch` forward FFTs sequentially on contiguous chunks of `data`.
    ///
    /// `data` must contain exactly `batch * n` values laid out as
    /// `[batch0..., batch1..., ...]`.
    pub fn forward_batch(&mut self, data: &mut [Complex<f64>], batch: usize) {
        let expected = batch
            .checked_mul(self.n)
            .expect("batch * n overflow in forward_batch");
        assert_eq!(
            data.len(),
            expected,
            "forward_batch expects data length {}, got {}",
            expected,
            data.len()
        );

        for chunk in data.chunks_exact_mut(self.n) {
            self.forward(chunk);
        }
    }

    fn execute(&mut self, data: &mut [Complex<f64>], direction: ffi::FFTDirection) {
        assert_eq!(
            data.len(),
            self.n,
            "FFT plan length mismatch: expected {}, got {}",
            self.n,
            data.len()
        );

        if let PlanBackend::MixedRadix3Pow2 {
            n1,
            setup_pow2,
            log2n1,
            stage_real,
            stage_imag,
            twiddle_forward_r1,
            twiddle_forward_r2,
            twiddle_inverse_r1,
            twiddle_inverse_r2,
        } = &mut self.backend
        {
            Self::execute_mixed_radix3_pow2(
                data,
                direction,
                *n1,
                *setup_pow2,
                *log2n1,
                &mut self.real_buffer,
                &mut self.imag_buffer,
                stage_real,
                stage_imag,
                twiddle_forward_r1,
                twiddle_forward_r2,
                twiddle_inverse_r1,
                twiddle_inverse_r2,
            );
            return;
        }

        enum StandardExec {
            Dft { setup: *mut c_void },
            Fft { setup: *mut c_void, log2n: usize },
        }

        let standard = match &self.backend {
            PlanBackend::Dft {
                setup_forward,
                setup_inverse,
            } => {
                let setup = if direction == ffi::kFFTDirection_Forward {
                    *setup_forward
                } else {
                    *setup_inverse
                };
                StandardExec::Dft { setup }
            }
            PlanBackend::Fft { setup, log2n } => StandardExec::Fft {
                setup: *setup,
                log2n: *log2n,
            },
            PlanBackend::MixedRadix3Pow2 { .. } => {
                unreachable!("mixed-radix handled by early return")
            }
        };

        match standard {
            StandardExec::Dft { setup } => {
                self.interleaved_to_split(data);
                assert!(!setup.is_null(), "vDSP DFT setup pointer is null");
                let setup = setup.cast::<ffi::vDSP_DFT_SetupStructD>();
                unsafe {
                    ffi::vDSP_DFT_ExecuteD(
                        setup as *const ffi::vDSP_DFT_SetupStructD,
                        self.real_buffer.as_ptr(),
                        self.imag_buffer.as_ptr(),
                        self.real_buffer.as_mut_ptr(),
                        self.imag_buffer.as_mut_ptr(),
                    );
                }
                self.split_to_interleaved(data);
            }
            StandardExec::Fft { setup, log2n } => {
                self.interleaved_to_split(data);
                Self::execute_pow2_in_place(
                    setup,
                    log2n,
                    &mut self.real_buffer[..self.n],
                    &mut self.imag_buffer[..self.n],
                    direction,
                );
                self.split_to_interleaved(data);
            }
        }
    }

    fn interleaved_to_split(&mut self, data: &[Complex<f64>]) {
        let split = self.split_complex_mut();
        unsafe {
            ffi::vDSP_ctozD(
                data.as_ptr().cast::<ffi::DSPDoubleComplex>(),
                2,
                &split as *const ffi::DSPDoubleSplitComplex,
                1,
                self.n,
            );
        }
    }

    fn split_to_interleaved(&mut self, data: &mut [Complex<f64>]) {
        let split = self.split_complex_mut();
        unsafe {
            ffi::vDSP_ztocD(
                &split as *const ffi::DSPDoubleSplitComplex,
                1,
                data.as_mut_ptr().cast::<ffi::DSPDoubleComplex>(),
                2,
                self.n,
            );
        }
    }

    fn split_complex_mut(&mut self) -> ffi::DSPDoubleSplitComplex {
        ffi::DSPDoubleSplitComplex {
            realp: self.real_buffer.as_mut_ptr(),
            imagp: self.imag_buffer.as_mut_ptr(),
        }
    }

    fn execute_pow2_in_place(
        setup: *mut c_void,
        log2n: usize,
        real: &mut [f64],
        imag: &mut [f64],
        direction: ffi::FFTDirection,
    ) {
        assert!(!setup.is_null(), "vDSP FFT setup pointer is null");
        assert_eq!(real.len(), imag.len(), "real/imag work buffer mismatch");
        let split = ffi::DSPDoubleSplitComplex {
            realp: real.as_mut_ptr(),
            imagp: imag.as_mut_ptr(),
        };
        unsafe {
            ffi::vDSP_fft_zipD(
                setup.cast::<ffi::OpaqueFFTSetupD>(),
                &split as *const ffi::DSPDoubleSplitComplex,
                1,
                log2n,
                direction,
            );
        }
    }

    fn execute_mixed_radix3_pow2(
        data: &mut [Complex<f64>],
        direction: ffi::FFTDirection,
        n1: usize,
        setup_pow2: *mut c_void,
        log2n1: usize,
        work_real: &mut [f64],
        work_imag: &mut [f64],
        stage_real: &mut [f64],
        stage_imag: &mut [f64],
        twiddle_forward_r1: &[Complex<f64>],
        twiddle_forward_r2: &[Complex<f64>],
        twiddle_inverse_r1: &[Complex<f64>],
        twiddle_inverse_r2: &[Complex<f64>],
    ) {
        let n = data.len();
        assert_eq!(n, n1 * 3, "mixed-radix expects n == 3*n1");
        assert!(work_real.len() >= n1 && work_imag.len() >= n1);
        assert!(stage_real.len() >= n && stage_imag.len() >= n);
        assert_eq!(twiddle_forward_r1.len(), n1);
        assert_eq!(twiddle_forward_r2.len(), n1);
        assert_eq!(twiddle_inverse_r1.len(), n1);
        assert_eq!(twiddle_inverse_r2.len(), n1);

        for r in 0..3usize {
            for m in 0..n1 {
                let v = data[m * 3 + r];
                work_real[m] = v.re;
                work_imag[m] = v.im;
            }

            Self::execute_pow2_in_place(
                setup_pow2,
                log2n1,
                &mut work_real[..n1],
                &mut work_imag[..n1],
                direction,
            );

            let row = r * n1;
            for k in 0..n1 {
                stage_real[row + k] = work_real[k];
                stage_imag[row + k] = work_imag[k];
            }
        }

        let sqrt3_over_2 = 0.866_025_403_784_438_6_f64;
        let (tw_r1, tw_r2, omega, omega2) = if direction == ffi::kFFTDirection_Forward {
            (
                twiddle_forward_r1,
                twiddle_forward_r2,
                Complex::new(-0.5, -sqrt3_over_2),
                Complex::new(-0.5, sqrt3_over_2),
            )
        } else {
            (
                twiddle_inverse_r1,
                twiddle_inverse_r2,
                Complex::new(-0.5, sqrt3_over_2),
                Complex::new(-0.5, -sqrt3_over_2),
            )
        };

        for k in 0..n1 {
            let a0 = Complex::new(stage_real[k], stage_imag[k]);
            let a1 = Complex::new(stage_real[n1 + k], stage_imag[n1 + k]);
            let a2 = Complex::new(stage_real[2 * n1 + k], stage_imag[2 * n1 + k]);

            let b0 = a0;
            let b1 = a1 * tw_r1[k];
            let b2 = a2 * tw_r2[k];

            let c0 = b0 + b1 + b2;
            let c1 = b0 + b1 * omega + b2 * omega2;
            let c2 = b0 + b1 * omega2 + b2 * omega;

            data[k] = c0;
            data[k + n1] = c1;
            data[k + 2 * n1] = c2;
        }
    }
}

impl Drop for AccelerateFFT {
    fn drop(&mut self) {
        unsafe {
            match &mut self.backend {
                PlanBackend::Dft {
                    setup_forward,
                    setup_inverse,
                } => {
                    ffi::vDSP_DFT_DestroySetupD(
                        (*setup_forward).cast::<ffi::vDSP_DFT_SetupStructD>(),
                    );
                    ffi::vDSP_DFT_DestroySetupD(
                        (*setup_inverse).cast::<ffi::vDSP_DFT_SetupStructD>(),
                    );
                }
                PlanBackend::Fft { setup, .. } => {
                    ffi::vDSP_destroy_fftsetupD((*setup).cast::<ffi::OpaqueFFTSetupD>());
                }
                PlanBackend::MixedRadix3Pow2 { setup_pow2, .. } => {
                    ffi::vDSP_destroy_fftsetupD((*setup_pow2).cast::<ffi::OpaqueFFTSetupD>());
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
fn resolve_requested_workers(workers: i32) -> usize {
    assert!(workers != 0, "workers must not be zero");

    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1) as i32;

    let requested = if workers > 0 {
        workers
    } else if workers == -1 {
        available
    } else {
        available - ((-workers) - 1)
    };

    assert!(
        requested >= 1,
        "workers={} resolves to {}, but effective worker count must be at least 1",
        workers,
        requested
    );
    requested as usize
}

#[cfg(feature = "parallel")]
fn resolve_effective_workers(workers: i32, jobs: usize) -> usize {
    assert!(jobs > 0, "independent job count must be positive");
    resolve_requested_workers(workers).min(jobs)
}

#[cfg(feature = "parallel")]
static PARALLEL_POOL_CACHE: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> =
    OnceLock::new();

#[cfg(feature = "parallel")]
thread_local! {
    static TLS_FFT_CACHE: RefCell<HashMap<usize, AccelerateFFT>> = RefCell::new(HashMap::new());
    static TLS_2D_SCRATCH: RefCell<HashMap<(usize, usize), Vec<Complex<f64>>>> = RefCell::new(HashMap::new());
    static TLS_3D_SCRATCH: RefCell<HashMap<(usize, usize, usize), Scratch3dBuffers>> = RefCell::new(HashMap::new());
}

#[cfg(feature = "parallel")]
fn with_cached_fft<R>(n: usize, f: impl FnOnce(&mut AccelerateFFT) -> R) -> R {
    TLS_FFT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let fft = cache.entry(n).or_insert_with(|| AccelerateFFT::new(n));
        f(fft)
    })
}

#[cfg(feature = "parallel")]
#[derive(Default)]
struct Scratch3dBuffers {
    y_lines: Vec<Complex<f64>>,
    z_lines: Vec<Complex<f64>>,
}

#[cfg(feature = "parallel")]
fn with_2d_scratch<R>(width: usize, height: usize, f: impl FnOnce(&mut [Complex<f64>]) -> R) -> R {
    let len = width
        .checked_mul(height)
        .expect("width * height overflow while preparing 2D scratch");
    TLS_2D_SCRATCH.with(|cache| {
        let mut cache = cache.borrow_mut();
        let buf = cache
            .entry((width, height))
            .or_insert_with(|| vec![Complex::new(0.0, 0.0); len]);
        if buf.len() != len {
            buf.resize(len, Complex::new(0.0, 0.0));
        }
        f(buf.as_mut_slice())
    })
}

#[cfg(feature = "parallel")]
fn with_3d_scratch<R>(
    width: usize,
    height: usize,
    depth: usize,
    f: impl FnOnce(&mut [Complex<f64>], &mut [Complex<f64>]) -> R,
) -> R {
    let wh = width
        .checked_mul(height)
        .expect("width * height overflow while preparing 3D scratch");
    let len = wh
        .checked_mul(depth)
        .expect("width * height * depth overflow while preparing 3D scratch");
    TLS_3D_SCRATCH.with(|cache| {
        let mut cache = cache.borrow_mut();
        let scratch = cache.entry((width, height, depth)).or_default();
        if scratch.y_lines.len() != len {
            scratch.y_lines.resize(len, Complex::new(0.0, 0.0));
        }
        if scratch.z_lines.len() != len {
            scratch.z_lines.resize(len, Complex::new(0.0, 0.0));
        }
        f(
            scratch.y_lines.as_mut_slice(),
            scratch.z_lines.as_mut_slice(),
        )
    })
}

#[cfg(feature = "parallel")]
fn get_or_build_worker_pool(effective: usize) -> Arc<rayon::ThreadPool> {
    let cache = PARALLEL_POOL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(existing) = cache
        .lock()
        .unwrap_or_else(|_| panic!("thread pool cache mutex poisoned"))
        .get(&effective)
        .cloned()
    {
        return existing;
    }

    let built = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(effective)
            .build()
            .unwrap_or_else(|err| {
                panic!(
                    "failed to build rayon thread pool with {} workers: {}",
                    effective, err
                )
            }),
    );

    let mut guard = cache
        .lock()
        .unwrap_or_else(|_| panic!("thread pool cache mutex poisoned"));
    guard
        .entry(effective)
        .or_insert_with(|| built.clone())
        .clone()
}

#[cfg(feature = "parallel")]
fn with_worker_pool<R: Send>(workers: i32, jobs: usize, f: impl FnOnce(usize) -> R + Send) -> R {
    let effective = resolve_effective_workers(workers, jobs);
    if effective == 1 {
        return f(1);
    }
    let pool = get_or_build_worker_pool(effective);
    pool.install(|| f(effective))
}

#[cfg(feature = "parallel")]
fn lines_per_group(jobs: usize, effective: usize) -> usize {
    let target_tasks = effective.saturating_mul(4).max(1);
    jobs.div_ceil(target_tasks).max(1)
}

#[cfg(feature = "parallel")]
fn process_contiguous_lines(
    data: &mut [Complex<f64>],
    line_len: usize,
    jobs: usize,
    workers: i32,
    direction: ffi::FFTDirection,
) {
    assert!(line_len > 0, "line_len must be positive");
    assert!(jobs > 0, "jobs must be positive");
    assert_eq!(
        data.len(),
        line_len
            .checked_mul(jobs)
            .expect("line_len * jobs overflow in process_contiguous_lines"),
        "process_contiguous_lines expects exactly jobs * line_len values",
    );

    with_worker_pool(workers, jobs, |effective| {
        if effective == 1 {
            with_cached_fft(line_len, |fft| {
                for line in data.chunks_exact_mut(line_len) {
                    if direction == ffi::kFFTDirection_Forward {
                        fft.forward(line);
                    } else {
                        fft.inverse(line);
                    }
                }
            });
            return;
        }

        let grouped_lines = lines_per_group(jobs, effective);
        let elems_per_group = line_len
            .checked_mul(grouped_lines)
            .expect("line_len * grouped_lines overflow in process_contiguous_lines");

        data.par_chunks_mut(elems_per_group).for_each(|group| {
            with_cached_fft(line_len, |fft| {
                for line in group.chunks_exact_mut(line_len) {
                    if direction == ffi::kFFTDirection_Forward {
                        fft.forward(line);
                    } else {
                        fft.inverse(line);
                    }
                }
            });
        });
    });
}

/// Parallel batch 1D forward FFTs with SciPy-like worker semantics.
///
/// `data` is interpreted as `batch` contiguous transforms of length `n`.
/// Available only when built with `--features parallel`.
#[cfg(feature = "parallel")]
pub fn fft_batch_with_workers(data: &mut [Complex<f64>], n: usize, batch: usize, workers: i32) {
    assert!(n > 0, "n must be greater than zero");
    let expected = batch
        .checked_mul(n)
        .expect("batch * n overflow in fft_batch_with_workers");
    assert_eq!(
        data.len(),
        expected,
        "fft_batch_with_workers expects data length {}, got {}",
        expected,
        data.len()
    );
    assert!(batch > 0, "batch must be greater than zero");
    process_contiguous_lines(data, n, batch, workers, ffi::kFFTDirection_Forward);
}

/// Parallel batch 1D inverse FFTs with SciPy-like worker semantics.
///
/// `data` is interpreted as `batch` contiguous transforms of length `n`.
/// Available only when built with `--features parallel`.
#[cfg(feature = "parallel")]
pub fn ifft_batch_with_workers(data: &mut [Complex<f64>], n: usize, batch: usize, workers: i32) {
    assert!(n > 0, "n must be greater than zero");
    let expected = batch
        .checked_mul(n)
        .expect("batch * n overflow in ifft_batch_with_workers");
    assert_eq!(
        data.len(),
        expected,
        "ifft_batch_with_workers expects data length {}, got {}",
        expected,
        data.len()
    );
    assert!(batch > 0, "batch must be greater than zero");
    process_contiguous_lines(data, n, batch, workers, ffi::kFFTDirection_Inverse);
}

/// Parallel 2D forward FFT with SciPy-like worker semantics.
///
/// Available only when built with `--features parallel`.
#[cfg(feature = "parallel")]
fn fft2_forward_with_workers_impl(
    data: &mut [Complex<f64>],
    width: usize,
    height: usize,
    workers: i32,
) {
    assert!(width > 0, "width must be greater than zero");
    assert!(height > 0, "height must be greater than zero");

    let expected = width
        .checked_mul(height)
        .expect("width * height overflow in fft2_with_workers");
    assert_eq!(
        data.len(),
        expected,
        "fft2_with_workers expects data length {}, got {}",
        expected,
        data.len()
    );

    process_contiguous_lines(data, width, height, workers, ffi::kFFTDirection_Forward);

    with_2d_scratch(width, height, |columns| {
        for x in 0..width {
            let line = x * height;
            for y in 0..height {
                columns[line + y] = data[y * width + x];
            }
        }

        process_contiguous_lines(columns, height, width, workers, ffi::kFFTDirection_Forward);

        for x in 0..width {
            let line = x * height;
            for y in 0..height {
                data[y * width + x] = columns[line + y];
            }
        }
    });
}

/// Parallel 2D inverse FFT with SciPy-like worker semantics.
///
/// Available only when built with `--features parallel`.
#[cfg(feature = "parallel")]
fn ifft2_with_workers_impl(data: &mut [Complex<f64>], width: usize, height: usize, workers: i32) {
    assert!(width > 0, "width must be greater than zero");
    assert!(height > 0, "height must be greater than zero");

    let expected = width
        .checked_mul(height)
        .expect("width * height overflow in ifft2_with_workers");
    assert_eq!(
        data.len(),
        expected,
        "ifft2_with_workers expects data length {}, got {}",
        expected,
        data.len()
    );

    process_contiguous_lines(data, width, height, workers, ffi::kFFTDirection_Inverse);

    with_2d_scratch(width, height, |columns| {
        for x in 0..width {
            let line = x * height;
            for y in 0..height {
                columns[line + y] = data[y * width + x];
            }
        }

        process_contiguous_lines(columns, height, width, workers, ffi::kFFTDirection_Inverse);

        for x in 0..width {
            let line = x * height;
            for y in 0..height {
                data[y * width + x] = columns[line + y];
            }
        }
    });
}

/// Parallel 3D forward FFT with SciPy-like worker semantics.
///
/// Available only when built with `--features parallel`.
#[cfg(feature = "parallel")]
fn fft3_with_workers_impl(
    data: &mut [Complex<f64>],
    width: usize,
    height: usize,
    depth: usize,
    workers: i32,
) {
    assert!(width > 0, "width must be greater than zero");
    assert!(height > 0, "height must be greater than zero");
    assert!(depth > 0, "depth must be greater than zero");

    let wh = width
        .checked_mul(height)
        .expect("width * height overflow in fft3_with_workers");
    let expected = wh
        .checked_mul(depth)
        .expect("width * height * depth overflow in fft3_with_workers");
    assert_eq!(
        data.len(),
        expected,
        "fft3_with_workers expects data length {}, got {}",
        expected,
        data.len()
    );

    process_contiguous_lines(
        data,
        width,
        height * depth,
        workers,
        ffi::kFFTDirection_Forward,
    );

    with_3d_scratch(width, height, depth, |y_lines, z_lines| {
        for z in 0..depth {
            let z_base = z * wh;
            for x in 0..width {
                let line = (z * width + x) * height;
                for y in 0..height {
                    y_lines[line + y] = data[z_base + y * width + x];
                }
            }
        }

        process_contiguous_lines(
            y_lines,
            height,
            width * depth,
            workers,
            ffi::kFFTDirection_Forward,
        );

        for z in 0..depth {
            let z_base = z * wh;
            for x in 0..width {
                let line = (z * width + x) * height;
                for y in 0..height {
                    data[z_base + y * width + x] = y_lines[line + y];
                }
            }
        }

        for y in 0..height {
            for x in 0..width {
                let line = (y * width + x) * depth;
                for z in 0..depth {
                    z_lines[line + z] = data[z * wh + y * width + x];
                }
            }
        }

        process_contiguous_lines(
            z_lines,
            depth,
            width * height,
            workers,
            ffi::kFFTDirection_Forward,
        );

        for y in 0..height {
            for x in 0..width {
                let line = (y * width + x) * depth;
                for z in 0..depth {
                    data[z * wh + y * width + x] = z_lines[line + z];
                }
            }
        }
    });
}

/// Parallel 3D inverse FFT with SciPy-like worker semantics.
///
/// Available only when built with `--features parallel`.
#[cfg(feature = "parallel")]
fn ifft3_with_workers_impl(
    data: &mut [Complex<f64>],
    width: usize,
    height: usize,
    depth: usize,
    workers: i32,
) {
    assert!(width > 0, "width must be greater than zero");
    assert!(height > 0, "height must be greater than zero");
    assert!(depth > 0, "depth must be greater than zero");

    let wh = width
        .checked_mul(height)
        .expect("width * height overflow in ifft3_with_workers");
    let expected = wh
        .checked_mul(depth)
        .expect("width * height * depth overflow in ifft3_with_workers");
    assert_eq!(
        data.len(),
        expected,
        "ifft3_with_workers expects data length {}, got {}",
        expected,
        data.len()
    );

    process_contiguous_lines(
        data,
        width,
        height * depth,
        workers,
        ffi::kFFTDirection_Inverse,
    );

    with_3d_scratch(width, height, depth, |y_lines, z_lines| {
        for z in 0..depth {
            let z_base = z * wh;
            for x in 0..width {
                let line = (z * width + x) * height;
                for y in 0..height {
                    y_lines[line + y] = data[z_base + y * width + x];
                }
            }
        }

        process_contiguous_lines(
            y_lines,
            height,
            width * depth,
            workers,
            ffi::kFFTDirection_Inverse,
        );

        for z in 0..depth {
            let z_base = z * wh;
            for x in 0..width {
                let line = (z * width + x) * height;
                for y in 0..height {
                    data[z_base + y * width + x] = y_lines[line + y];
                }
            }
        }

        for y in 0..height {
            for x in 0..width {
                let line = (y * width + x) * depth;
                for z in 0..depth {
                    z_lines[line + z] = data[z * wh + y * width + x];
                }
            }
        }

        process_contiguous_lines(
            z_lines,
            depth,
            width * height,
            workers,
            ffi::kFFTDirection_Inverse,
        );

        for y in 0..height {
            for x in 0..width {
                let line = (y * width + x) * depth;
                for z in 0..depth {
                    data[z * wh + y * width + x] = z_lines[line + z];
                }
            }
        }
    });
}

// Internal 2D forward transform using row-then-column decomposition.
fn fft2_forward_impl(data: &mut [Complex<f64>], width: usize, height: usize) {
    assert!(width > 0, "width must be greater than zero");
    assert!(height > 0, "height must be greater than zero");

    let expected = width
        .checked_mul(height)
        .expect("width * height overflow in fft2");
    assert_eq!(
        data.len(),
        expected,
        "fft2 expects data length {}, got {}",
        expected,
        data.len()
    );

    let mut row_fft = AccelerateFFT::new(width);
    for row in data.chunks_exact_mut(width) {
        row_fft.forward(row);
    }

    let mut col_fft = AccelerateFFT::new(height);
    let mut column = vec![Complex::new(0.0, 0.0); height];
    for x in 0..width {
        for y in 0..height {
            column[y] = data[y * width + x];
        }
        col_fft.forward(&mut column);
        for y in 0..height {
            data[y * width + x] = column[y];
        }
    }
}

/// Simple 1D forward FFT API, analogous to `scipy.fft.fft`.
///
/// Builds a temporary plan for `data.len()` and executes one forward transform.
/// For repeated execution, prefer explicit `AccelerateFFT` plan reuse.
pub fn fft(data: &mut [Complex<f64>]) {
    assert!(!data.is_empty(), "data must not be empty");
    let mut fft = AccelerateFFT::new(data.len());
    fft.forward(data);
}

/// Simple 1D inverse FFT API, analogous to `scipy.fft.ifft`.
///
/// Builds a temporary plan for `data.len()` and executes one inverse transform.
/// This inverse is unnormalized (FFTW-style).
/// For repeated execution, prefer explicit `AccelerateFFT` plan reuse.
pub fn ifft(data: &mut [Complex<f64>]) {
    assert!(!data.is_empty(), "data must not be empty");
    let mut fft = AccelerateFFT::new(data.len());
    fft.inverse(data);
}

/// Simple 2D forward FFT API, analogous to `scipy.fft.fft2`.
pub fn fft2(data: &mut [Complex<f64>], width: usize, height: usize) {
    fft2_forward_impl(data, width, height);
}

/// Simple 2D inverse FFT API, analogous to `scipy.fft.ifft2`.
pub fn ifft2(data: &mut [Complex<f64>], width: usize, height: usize) {
    ifft2_impl(data, width, height);
}

/// Simple 3D forward FFT API, analogous to `scipy.fft.fftn` for rank-3 input.
pub fn fft3(data: &mut [Complex<f64>], width: usize, height: usize, depth: usize) {
    fft3_impl(data, width, height, depth);
}

/// Simple 3D inverse FFT API, analogous to `scipy.fft.ifftn` for rank-3 input.
pub fn ifft3(data: &mut [Complex<f64>], width: usize, height: usize, depth: usize) {
    ifft3_impl(data, width, height, depth);
}

// Internal 2D inverse transform using row-then-column decomposition.
fn ifft2_impl(data: &mut [Complex<f64>], width: usize, height: usize) {
    assert!(width > 0, "width must be greater than zero");
    assert!(height > 0, "height must be greater than zero");

    let expected = width
        .checked_mul(height)
        .expect("width * height overflow in ifft2");
    assert_eq!(
        data.len(),
        expected,
        "ifft2 expects data length {}, got {}",
        expected,
        data.len()
    );

    let mut row_fft = AccelerateFFT::new(width);
    for row in data.chunks_exact_mut(width) {
        row_fft.inverse(row);
    }

    let mut col_fft = AccelerateFFT::new(height);
    let mut column = vec![Complex::new(0.0, 0.0); height];
    for x in 0..width {
        for y in 0..height {
            column[y] = data[y * width + x];
        }
        col_fft.inverse(&mut column);
        for y in 0..height {
            data[y * width + x] = column[y];
        }
    }
}

// Internal 3D forward transform using separable 1D transforms over x, y, and z.
fn fft3_impl(data: &mut [Complex<f64>], width: usize, height: usize, depth: usize) {
    assert!(width > 0, "width must be greater than zero");
    assert!(height > 0, "height must be greater than zero");
    assert!(depth > 0, "depth must be greater than zero");

    let wh = width
        .checked_mul(height)
        .expect("width * height overflow in fft3");
    let expected = wh
        .checked_mul(depth)
        .expect("width * height * depth overflow in fft3");
    assert_eq!(
        data.len(),
        expected,
        "fft3 expects data length {}, got {}",
        expected,
        data.len()
    );

    let mut x_fft = AccelerateFFT::new(width);
    for z in 0..depth {
        let z_base = z * wh;
        for y in 0..height {
            let start = z_base + y * width;
            let end = start + width;
            x_fft.forward(&mut data[start..end]);
        }
    }

    let mut y_fft = AccelerateFFT::new(height);
    let mut y_line = vec![Complex::new(0.0, 0.0); height];
    for z in 0..depth {
        let z_base = z * wh;
        for x in 0..width {
            for y in 0..height {
                y_line[y] = data[z_base + y * width + x];
            }
            y_fft.forward(&mut y_line);
            for y in 0..height {
                data[z_base + y * width + x] = y_line[y];
            }
        }
    }

    let mut z_fft = AccelerateFFT::new(depth);
    let mut z_line = vec![Complex::new(0.0, 0.0); depth];
    for y in 0..height {
        for x in 0..width {
            for z in 0..depth {
                z_line[z] = data[z * wh + y * width + x];
            }
            z_fft.forward(&mut z_line);
            for z in 0..depth {
                data[z * wh + y * width + x] = z_line[z];
            }
        }
    }
}

// Internal 3D inverse transform using separable 1D transforms over x, y, and z.
fn ifft3_impl(data: &mut [Complex<f64>], width: usize, height: usize, depth: usize) {
    assert!(width > 0, "width must be greater than zero");
    assert!(height > 0, "height must be greater than zero");
    assert!(depth > 0, "depth must be greater than zero");

    let wh = width
        .checked_mul(height)
        .expect("width * height overflow in ifft3");
    let expected = wh
        .checked_mul(depth)
        .expect("width * height * depth overflow in ifft3");
    assert_eq!(
        data.len(),
        expected,
        "ifft3 expects data length {}, got {}",
        expected,
        data.len()
    );

    let mut x_fft = AccelerateFFT::new(width);
    for z in 0..depth {
        let z_base = z * wh;
        for y in 0..height {
            let start = z_base + y * width;
            let end = start + width;
            x_fft.inverse(&mut data[start..end]);
        }
    }

    let mut y_fft = AccelerateFFT::new(height);
    let mut y_line = vec![Complex::new(0.0, 0.0); height];
    for z in 0..depth {
        let z_base = z * wh;
        for x in 0..width {
            for y in 0..height {
                y_line[y] = data[z_base + y * width + x];
            }
            y_fft.inverse(&mut y_line);
            for y in 0..height {
                data[z_base + y * width + x] = y_line[y];
            }
        }
    }

    let mut z_fft = AccelerateFFT::new(depth);
    let mut z_line = vec![Complex::new(0.0, 0.0); depth];
    for y in 0..height {
        for x in 0..width {
            for z in 0..depth {
                z_line[z] = data[z * wh + y * width + x];
            }
            z_fft.inverse(&mut z_line);
            for z in 0..depth {
                data[z * wh + y * width + x] = z_line[z];
            }
        }
    }
}

#[cfg(feature = "parallel")]
/// Simple 1D forward FFT API with SciPy-like `workers` semantics.
pub fn fft_with_workers(data: &mut [Complex<f64>], workers: i32) {
    assert!(!data.is_empty(), "data must not be empty");
    fft_batch_with_workers(data, data.len(), 1, workers);
}

#[cfg(feature = "parallel")]
/// Simple 1D inverse FFT API with SciPy-like `workers` semantics.
pub fn ifft_with_workers(data: &mut [Complex<f64>], workers: i32) {
    assert!(!data.is_empty(), "data must not be empty");
    ifft_batch_with_workers(data, data.len(), 1, workers);
}

#[cfg(feature = "parallel")]
/// Simple 2D forward FFT API with SciPy-like `workers` semantics.
pub fn fft2_with_workers(data: &mut [Complex<f64>], width: usize, height: usize, workers: i32) {
    fft2_forward_with_workers_impl(data, width, height, workers);
}

#[cfg(feature = "parallel")]
/// Simple 2D inverse FFT API with SciPy-like `workers` semantics.
pub fn ifft2_with_workers(data: &mut [Complex<f64>], width: usize, height: usize, workers: i32) {
    ifft2_with_workers_impl(data, width, height, workers);
}

#[cfg(feature = "parallel")]
/// Simple 3D forward FFT API with SciPy-like `workers` semantics.
pub fn fft3_with_workers(
    data: &mut [Complex<f64>],
    width: usize,
    height: usize,
    depth: usize,
    workers: i32,
) {
    fft3_with_workers_impl(data, width, height, depth, workers);
}

#[cfg(feature = "parallel")]
/// Simple 3D inverse FFT API with SciPy-like `workers` semantics.
pub fn ifft3_with_workers(
    data: &mut [Complex<f64>],
    width: usize,
    height: usize,
    depth: usize,
    workers: i32,
) {
    ifft3_with_workers_impl(data, width, height, depth, workers);
}
