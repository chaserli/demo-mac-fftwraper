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
        pub fn vDSP_mtransD(
            a: *const f64,
            ia: vDSP_Stride,
            c: *mut f64,
            ic: vDSP_Stride,
            m: vDSP_Length,
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
        pub fn vDSP_fft_ziptD(
            setup: FFTSetupD,
            c: *const DSPDoubleSplitComplex,
            stride: vDSP_Stride,
            buffer: *const DSPDoubleSplitComplex,
            log2n: vDSP_Length,
            direction: FFTDirection,
        );
        pub fn vDSP_fftm_ziptD(
            setup: FFTSetupD,
            c: *const DSPDoubleSplitComplex,
            ic: vDSP_Stride,
            im: vDSP_Stride,
            buffer: *const DSPDoubleSplitComplex,
            log2n: vDSP_Length,
            m: vDSP_Length,
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
    DFT {
        setup_forward: *mut c_void,
        setup_inverse: *mut c_void,
    },
    FFT {
        setup: *mut c_void,
        log2n: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccelerateFFTError {
    InvalidLength,
    ForwardSetupFailed { n: usize },
    InverseSetupFailed { n: usize },
}

pub type AccelerateFftError = AccelerateFFTError;

impl fmt::Display for AccelerateFFTError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccelerateFFTError::InvalidLength => {
                write!(f, "FFT length must be greater than zero")
            }
            AccelerateFFTError::ForwardSetupFailed { n } => {
                write!(f, "failed to create forward vDSP DFT setup for length {n}")
            }
            AccelerateFFTError::InverseSetupFailed { n } => {
                write!(f, "failed to create inverse vDSP DFT setup for length {n}")
            }
        }
    }
}

impl Error for AccelerateFFTError {}

impl AccelerateFFT {
    /// Creates a reusable FFT plan for transforms of length `n`.
    ///
    /// Returns a typed error instead of panicking.
    ///
    /// The backend strategy is intentionally simple:
    /// - radix-2 lengths use `vDSP_fft_zipD`
    /// - everything else attempts `vDSP_DFT_*`
    ///
    /// This keeps the crate as a thin Accelerate wrapper rather than a
    /// benchmark-driven mini planner. `try_new` is therefore the honest way to
    /// probe whether a given non-radix-2 length is supported on the current
    /// Accelerate version / CPU combination.
    pub fn try_new(n: usize) -> Result<AccelerateFFT, AccelerateFFTError> {
        if n == 0 {
            return Err(AccelerateFFTError::InvalidLength);
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
                    backend: PlanBackend::FFT {
                        setup: setup_fft.cast::<c_void>(),
                        log2n,
                    },
                    real_buffer,
                    imag_buffer,
                });
            }
        }

        let setup_forward = unsafe {
            ffi::vDSP_DFT_zop_CreateSetupD(std::ptr::null_mut(), n, ffi::vDSP_DFT_FORWARD)
        };
        if setup_forward.is_null() {
            return Err(AccelerateFFTError::ForwardSetupFailed { n });
        }

        let setup_inverse =
            unsafe { ffi::vDSP_DFT_zop_CreateSetupD(setup_forward, n, ffi::vDSP_DFT_INVERSE) };
        if setup_inverse.is_null() {
            unsafe { ffi::vDSP_DFT_DestroySetupD(setup_forward) };
            return Err(AccelerateFFTError::InverseSetupFailed { n });
        }

        Ok(AccelerateFFT {
            n,
            backend: PlanBackend::DFT {
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

        Self::interleaved_to_split_slices(data, &mut self.real_buffer, &mut self.imag_buffer);
        match &mut self.backend {
            PlanBackend::DFT {
                setup_forward,
                setup_inverse,
            } => {
                let setup = if direction == ffi::kFFTDirection_Forward {
                    *setup_forward
                } else {
                    *setup_inverse
                };
                Self::execute_dft_in_place(
                    setup,
                    &mut self.real_buffer[..self.n],
                    &mut self.imag_buffer[..self.n],
                );
            }
            PlanBackend::FFT { setup, log2n } => {
                Self::execute_pow2_in_place(
                    *setup,
                    *log2n,
                    &mut self.real_buffer[..self.n],
                    &mut self.imag_buffer[..self.n],
                    direction,
                );
            }
        }
        Self::split_slices_to_interleaved(&self.real_buffer, &self.imag_buffer, data);
    }

    #[cfg(feature = "parallel")]
    fn execute_split_in_place(
        &mut self,
        real: &mut [f64],
        imag: &mut [f64],
        direction: ffi::FFTDirection,
    ) {
        assert_eq!(real.len(), self.n, "real split buffer length mismatch");
        assert_eq!(imag.len(), self.n, "imag split buffer length mismatch");
        match &mut self.backend {
            PlanBackend::DFT {
                setup_forward,
                setup_inverse,
            } => {
                let setup = if direction == ffi::kFFTDirection_Forward {
                    *setup_forward
                } else {
                    *setup_inverse
                };
                Self::execute_dft_in_place(setup, real, imag);
            }
            PlanBackend::FFT { setup, log2n } => {
                let split = ffi::DSPDoubleSplitComplex {
                    realp: real.as_mut_ptr(),
                    imagp: imag.as_mut_ptr(),
                };
                let temp = ffi::DSPDoubleSplitComplex {
                    realp: self.real_buffer.as_mut_ptr(),
                    imagp: self.imag_buffer.as_mut_ptr(),
                };
                unsafe {
                    ffi::vDSP_fft_ziptD(
                        setup.cast::<ffi::OpaqueFFTSetupD>(),
                        &split as *const ffi::DSPDoubleSplitComplex,
                        1,
                        &temp as *const ffi::DSPDoubleSplitComplex,
                        *log2n,
                        direction,
                    );
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    fn execute_split_batch_in_place(
        &mut self,
        real: &mut [f64],
        imag: &mut [f64],
        jobs: usize,
        direction: ffi::FFTDirection,
    ) {
        assert!(jobs > 0, "jobs must be positive");
        let expected = self
            .n
            .checked_mul(jobs)
            .expect("n * jobs overflow in execute_split_batch_in_place");
        assert_eq!(real.len(), expected, "real split buffer length mismatch");
        assert_eq!(imag.len(), expected, "imag split buffer length mismatch");

        match &mut self.backend {
            PlanBackend::DFT { .. } => {
                for (real_line, imag_line) in real
                    .chunks_exact_mut(self.n)
                    .zip(imag.chunks_exact_mut(self.n))
                {
                    self.execute_split_in_place(real_line, imag_line, direction);
                }
            }
            PlanBackend::FFT { setup, log2n } => {
                let split = ffi::DSPDoubleSplitComplex {
                    realp: real.as_mut_ptr(),
                    imagp: imag.as_mut_ptr(),
                };
                let temp = ffi::DSPDoubleSplitComplex {
                    realp: self.real_buffer.as_mut_ptr(),
                    imagp: self.imag_buffer.as_mut_ptr(),
                };
                unsafe {
                    ffi::vDSP_fftm_ziptD(
                        setup.cast::<ffi::OpaqueFFTSetupD>(),
                        &split as *const ffi::DSPDoubleSplitComplex,
                        1,
                        self.n as ffi::vDSP_Stride,
                        &temp as *const ffi::DSPDoubleSplitComplex,
                        *log2n,
                        jobs,
                        direction,
                    );
                }
            }
        }
    }

    fn execute_dft_in_place(setup: *mut c_void, real: &mut [f64], imag: &mut [f64]) {
        assert!(!setup.is_null(), "vDSP DFT setup pointer is null");
        let setup = setup.cast::<ffi::vDSP_DFT_SetupStructD>();
        unsafe {
            ffi::vDSP_DFT_ExecuteD(
                setup as *const ffi::vDSP_DFT_SetupStructD,
                real.as_ptr(),
                imag.as_ptr(),
                real.as_mut_ptr(),
                imag.as_mut_ptr(),
            );
        }
    }

    fn interleaved_to_split_slices(data: &[Complex<f64>], real: &mut [f64], imag: &mut [f64]) {
        assert_eq!(data.len(), real.len(), "real split buffer length mismatch");
        assert_eq!(data.len(), imag.len(), "imag split buffer length mismatch");
        let split = ffi::DSPDoubleSplitComplex {
            realp: real.as_mut_ptr(),
            imagp: imag.as_mut_ptr(),
        };
        unsafe {
            ffi::vDSP_ctozD(
                data.as_ptr().cast::<ffi::DSPDoubleComplex>(),
                2,
                &split as *const ffi::DSPDoubleSplitComplex,
                1,
                data.len(),
            );
        }
    }

    #[cfg(feature = "parallel")]
    fn interleaved_to_split_slices_parallel(
        data: &[Complex<f64>],
        real: &mut [f64],
        imag: &mut [f64],
        effective: usize,
    ) {
        assert_eq!(data.len(), real.len(), "real split buffer length mismatch");
        assert_eq!(data.len(), imag.len(), "imag split buffer length mismatch");

        const MIN_PARALLEL_ELEMS: usize = 1 << 16;
        if effective == 1 || data.len() < MIN_PARALLEL_ELEMS {
            Self::interleaved_to_split_slices(data, real, imag);
            return;
        }

        let stage_parallelism = memory_bound_parallelism(effective, data.len());
        let elems_per_group = data.len().div_ceil(stage_parallelism).max(1);
        data.par_chunks(elems_per_group)
            .zip(real.par_chunks_mut(elems_per_group))
            .zip(imag.par_chunks_mut(elems_per_group))
            .for_each(|((data_group, real_group), imag_group)| {
                Self::interleaved_to_split_slices(data_group, real_group, imag_group);
            });
    }

    fn split_slices_to_interleaved(real: &[f64], imag: &[f64], data: &mut [Complex<f64>]) {
        assert_eq!(data.len(), real.len(), "real split buffer length mismatch");
        assert_eq!(data.len(), imag.len(), "imag split buffer length mismatch");
        let split = ffi::DSPDoubleSplitComplex {
            realp: real.as_ptr() as *mut f64,
            imagp: imag.as_ptr() as *mut f64,
        };
        unsafe {
            ffi::vDSP_ztocD(
                &split as *const ffi::DSPDoubleSplitComplex,
                1,
                data.as_mut_ptr().cast::<ffi::DSPDoubleComplex>(),
                2,
                data.len(),
            );
        }
    }

    #[cfg(feature = "parallel")]
    fn split_slices_to_interleaved_parallel(
        real: &[f64],
        imag: &[f64],
        data: &mut [Complex<f64>],
        effective: usize,
    ) {
        assert_eq!(data.len(), real.len(), "real split buffer length mismatch");
        assert_eq!(data.len(), imag.len(), "imag split buffer length mismatch");

        const MIN_PARALLEL_ELEMS: usize = 1 << 16;
        if effective == 1 || data.len() < MIN_PARALLEL_ELEMS {
            Self::split_slices_to_interleaved(real, imag, data);
            return;
        }

        let stage_parallelism = memory_bound_parallelism(effective, data.len());
        let elems_per_group = data.len().div_ceil(stage_parallelism).max(1);
        real.par_chunks(elems_per_group)
            .zip(imag.par_chunks(elems_per_group))
            .zip(data.par_chunks_mut(elems_per_group))
            .for_each(|((real_group, imag_group), data_group)| {
                Self::split_slices_to_interleaved(real_group, imag_group, data_group);
            });
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

}

impl Drop for AccelerateFFT {
    fn drop(&mut self) {
        unsafe {
            match &mut self.backend {
                PlanBackend::DFT {
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
                PlanBackend::FFT { setup, .. } => {
                    ffi::vDSP_destroy_fftsetupD((*setup).cast::<ffi::OpaqueFFTSetupD>());
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
fn transpose_matrix(input: &[f64], output: &mut [f64], rows: usize, cols: usize) {
    let expected = rows
        .checked_mul(cols)
        .expect("rows * cols overflow in transpose_matrix");
    assert_eq!(input.len(), expected, "transpose input length mismatch");
    assert_eq!(output.len(), expected, "transpose output length mismatch");
    unsafe {
        ffi::vDSP_mtransD(input.as_ptr(), 1, output.as_mut_ptr(), 1, cols, rows);
    }
}

#[cfg(feature = "parallel")]
fn memory_bound_parallelism(effective: usize, elems: usize) -> usize {
    if effective <= 2 {
        effective
    } else if elems < (1 << 20) {
        effective.min(4)
    } else if elems < (1 << 22) {
        effective.min(6)
    } else {
        effective.min(8)
    }
}

#[cfg(feature = "parallel")]
fn transpose_matrix_parallel_impl(
    input: &[f64],
    output: &mut [f64],
    rows: usize,
    cols: usize,
    effective: usize,
) {
    let expected = rows
        .checked_mul(cols)
        .expect("rows * cols overflow in transpose_matrix_parallel");
    assert_eq!(input.len(), expected, "transpose input length mismatch");
    assert_eq!(output.len(), expected, "transpose output length mismatch");

    const TILE_ROWS: usize = 32;
    const TILE_COLS: usize = 32;

    if effective == 1 || expected < (1 << 15) {
        transpose_matrix(input, output, rows, cols);
        return;
    }

    let stage_parallelism = memory_bound_parallelism(effective, expected);
    let cols_per_group = cols.div_ceil(stage_parallelism).max(1);
    let elems_per_group = rows
        .checked_mul(cols_per_group)
        .expect("rows * cols_per_group overflow in transpose_matrix_parallel");

    output
        .par_chunks_mut(elems_per_group)
        .enumerate()
        .for_each(|(group_idx, output_group)| {
            let col_start = group_idx * cols_per_group;
            let group_cols = output_group.len() / rows;

            for local_col_block in (0..group_cols).step_by(TILE_COLS) {
                let local_col_end = (local_col_block + TILE_COLS).min(group_cols);
                for row_block in (0..rows).step_by(TILE_ROWS) {
                    let row_end = (row_block + TILE_ROWS).min(rows);
                    for local_col in local_col_block..local_col_end {
                        let src_col = col_start + local_col;
                        let out_base = local_col * rows + row_block;
                        for row in row_block..row_end {
                            output_group[out_base + (row - row_block)] =
                                input[row * cols + src_col];
                        }
                    }
                }
            }
        });
}

struct FFT2Plan {
    width: usize,
    height: usize,
    total: usize,
    real: Vec<f64>,
    imag: Vec<f64>,
    trans_real: Vec<f64>,
    trans_imag: Vec<f64>,
}

#[cfg(feature = "parallel")]
impl FFT2Plan {
    fn new(width: usize, height: usize) -> Self {
        assert!(width > 0, "width must be greater than zero");
        assert!(height > 0, "height must be greater than zero");
        let total = width
            .checked_mul(height)
            .expect("width * height overflow in FFT2Plan::new");
        Self {
            width,
            height,
            total,
            real: vec![0.0; total],
            imag: vec![0.0; total],
            trans_real: vec![0.0; total],
            trans_imag: vec![0.0; total],
        }
    }

    #[cfg(feature = "parallel")]
    fn execute_with_workers(
        &mut self,
        data: &mut [Complex<f64>],
        workers: i32,
        direction: ffi::FFTDirection,
    ) {
        assert_eq!(data.len(), self.total, "FFT2Plan input length mismatch");
        let width = self.width;
        let height = self.height;
        let real = &mut self.real;
        let imag = &mut self.imag;
        let trans_real = &mut self.trans_real;
        let trans_imag = &mut self.trans_imag;
        with_worker_pool(workers, self.width.max(self.height), |effective| {
            AccelerateFFT::interleaved_to_split_slices_parallel(data, real, imag, effective);
            process_split_lines_impl(real, imag, width, height, effective, direction);
            transpose_matrix_parallel_impl(real, trans_real, height, width, effective);
            transpose_matrix_parallel_impl(imag, trans_imag, height, width, effective);
            process_split_lines_impl(trans_real, trans_imag, height, width, effective, direction);
            transpose_matrix_parallel_impl(trans_real, real, width, height, effective);
            transpose_matrix_parallel_impl(trans_imag, imag, width, height, effective);
            AccelerateFFT::split_slices_to_interleaved_parallel(real, imag, data, effective);
        });
    }
}

#[cfg(feature = "parallel")]
struct FFT3Plan {
    width: usize,
    height: usize,
    depth: usize,
    plane_len: usize,
    total: usize,
    real: Vec<f64>,
    imag: Vec<f64>,
}

#[cfg(feature = "parallel")]
impl FFT3Plan {
    fn new(width: usize, height: usize, depth: usize) -> Self {
        assert!(width > 0, "width must be greater than zero");
        assert!(height > 0, "height must be greater than zero");
        assert!(depth > 0, "depth must be greater than zero");
        let plane_len = width
            .checked_mul(height)
            .expect("width * height overflow in FFT3Plan::new");
        let total = plane_len
            .checked_mul(depth)
            .expect("width * height * depth overflow in FFT3Plan::new");
        Self {
            width,
            height,
            depth,
            plane_len,
            total,
            real: vec![0.0; total],
            imag: vec![0.0; total],
        }
    }

    #[cfg(feature = "parallel")]
    fn execute_with_workers(
        &mut self,
        data: &mut [Complex<f64>],
        workers: i32,
        direction: ffi::FFTDirection,
    ) {
        assert_eq!(data.len(), self.total, "FFT3Plan input length mismatch");
        let width = self.width;
        let height = self.height;
        let depth = self.depth;
        let plane_len = self.plane_len;
        let real = &mut self.real;
        let imag = &mut self.imag;
        let max_jobs = (self.height * self.depth)
            .max(self.width * self.depth)
            .max(self.plane_len);
        with_worker_pool(workers, max_jobs, |effective| {
            AccelerateFFT::interleaved_to_split_slices_parallel(data, real, imag, effective);
            process_split_lines_impl(real, imag, width, height * depth, effective, direction);
            execute_y_stage_blocked_split(real, imag, width, height, depth, effective, direction);
            execute_z_stage_blocked_split(real, imag, plane_len, depth, effective, direction);
            AccelerateFFT::split_slices_to_interleaved_parallel(real, imag, data, effective);
        });
    }
}

#[cfg(feature = "parallel")]
fn scratch_jobs_for_line_len(line_len: usize, max_jobs: usize) -> usize {
    const TARGET_SCRATCH_ELEMS: usize = 16_384;
    const MAX_BLOCK_JOBS: usize = 256;
    ((TARGET_SCRATCH_ELEMS / line_len).max(1))
        .min(max_jobs)
        .min(MAX_BLOCK_JOBS)
}

#[cfg(feature = "parallel")]
fn gather_columns_into_scratch(
    src_real: &[f64],
    src_imag: &[f64],
    width: usize,
    height: usize,
    col_start: usize,
    cols_in_block: usize,
    scratch_real: &mut [f64],
    scratch_imag: &mut [f64],
) {
    for local_col in 0..cols_in_block {
        let x = col_start + local_col;
        let dst_real = &mut scratch_real[local_col * height..(local_col + 1) * height];
        let dst_imag = &mut scratch_imag[local_col * height..(local_col + 1) * height];
        for y in 0..height {
            let src_idx = y * width + x;
            dst_real[y] = src_real[src_idx];
            dst_imag[y] = src_imag[src_idx];
        }
    }
}

#[cfg(feature = "parallel")]
fn scatter_columns_from_scratch(
    dst_real: &mut [f64],
    dst_imag: &mut [f64],
    width: usize,
    height: usize,
    col_start: usize,
    cols_in_block: usize,
    scratch_real: &[f64],
    scratch_imag: &[f64],
) {
    for local_col in 0..cols_in_block {
        let x = col_start + local_col;
        let src_real = &scratch_real[local_col * height..(local_col + 1) * height];
        let src_imag = &scratch_imag[local_col * height..(local_col + 1) * height];
        for y in 0..height {
            let dst_idx = y * width + x;
            dst_real[dst_idx] = src_real[y];
            dst_imag[dst_idx] = src_imag[y];
        }
    }
}

#[cfg(feature = "parallel")]
fn execute_y_stage_blocked_split(
    real: &mut [f64],
    imag: &mut [f64],
    width: usize,
    height: usize,
    depth: usize,
    effective: usize,
    direction: ffi::FFTDirection,
) {
    let plane_len = width
        .checked_mul(height)
        .expect("width * height overflow in execute_y_stage_blocked_split");
    let total = plane_len
        .checked_mul(depth)
        .expect("plane_len * depth overflow in execute_y_stage_blocked_split");
    assert_eq!(real.len(), total, "real split buffer length mismatch");
    assert_eq!(imag.len(), total, "imag split buffer length mismatch");

    let block_cols = scratch_jobs_for_line_len(height, width);
    let blocks_per_plane = width.div_ceil(block_cols);
    let jobs = depth
        .checked_mul(blocks_per_plane)
        .expect("depth * blocks_per_plane overflow in execute_y_stage_blocked_split");

    if effective == 1 {
        for z in 0..depth {
            let plane_offset = z * plane_len;
            let real_plane = &mut real[plane_offset..plane_offset + plane_len];
            let imag_plane = &mut imag[plane_offset..plane_offset + plane_len];
            for block_idx in 0..blocks_per_plane {
                let col_start = block_idx * block_cols;
                let cols_in_block = (width - col_start).min(block_cols);
                let used = cols_in_block * height;
                with_cached_split_scratch(block_cols * height, |scratch_real, scratch_imag| {
                    let scratch_real = &mut scratch_real[..used];
                    let scratch_imag = &mut scratch_imag[..used];
                    gather_columns_into_scratch(
                        real_plane,
                        imag_plane,
                        width,
                        height,
                        col_start,
                        cols_in_block,
                        scratch_real,
                        scratch_imag,
                    );
                    with_cached_fft(height, |fft| {
                        fft.execute_split_batch_in_place(
                            scratch_real, scratch_imag, cols_in_block, direction,
                        );
                    });
                    scatter_columns_from_scratch(
                        real_plane,
                        imag_plane,
                        width,
                        height,
                        col_start,
                        cols_in_block,
                        scratch_real,
                        scratch_imag,
                    );
                });
            }
        }
        return;
    }

    let real_addr = real.as_mut_ptr() as usize;
    let imag_addr = imag.as_mut_ptr() as usize;
    (0..jobs).into_par_iter().for_each(|job_idx| {
        let real_ptr = real_addr as *mut f64;
        let imag_ptr = imag_addr as *mut f64;
        let z = job_idx / blocks_per_plane;
        let block_idx = job_idx % blocks_per_plane;
        let plane_offset = z * plane_len;
        let col_start = block_idx * block_cols;
        let cols_in_block = (width - col_start).min(block_cols);
        let used = cols_in_block * height;

        with_cached_split_scratch(block_cols * height, |scratch_real, scratch_imag| {
            let scratch_real = &mut scratch_real[..used];
            let scratch_imag = &mut scratch_imag[..used];

            for local_col in 0..cols_in_block {
                let x = col_start + local_col;
                let dst_real_col =
                    &mut scratch_real[local_col * height..(local_col + 1) * height];
                let dst_imag_col =
                    &mut scratch_imag[local_col * height..(local_col + 1) * height];
                for y in 0..height {
                    let idx = plane_offset + y * width + x;
                    unsafe {
                        dst_real_col[y] = *real_ptr.add(idx);
                        dst_imag_col[y] = *imag_ptr.add(idx);
                    }
                }
            }

            with_cached_fft(height, |fft| {
                fft.execute_split_batch_in_place(
                    scratch_real, scratch_imag, cols_in_block, direction,
                );
            });

            for local_col in 0..cols_in_block {
                let x = col_start + local_col;
                let src_real_col =
                    &scratch_real[local_col * height..(local_col + 1) * height];
                let src_imag_col =
                    &scratch_imag[local_col * height..(local_col + 1) * height];
                for y in 0..height {
                    let idx = plane_offset + y * width + x;
                    unsafe {
                        *real_ptr.add(idx) = src_real_col[y];
                        *imag_ptr.add(idx) = src_imag_col[y];
                    }
                }
            }
        });
    });
}

#[cfg(feature = "parallel")]
fn execute_z_stage_blocked_split(
    real: &mut [f64],
    imag: &mut [f64],
    plane_len: usize,
    depth: usize,
    effective: usize,
    direction: ffi::FFTDirection,
) {
    let total = plane_len
        .checked_mul(depth)
        .expect("plane_len * depth overflow in execute_z_stage_blocked_split");
    assert_eq!(real.len(), total, "real split buffer length mismatch");
    assert_eq!(imag.len(), total, "imag split buffer length mismatch");

    let block_lines = scratch_jobs_for_line_len(depth, plane_len);
    let blocks = plane_len.div_ceil(block_lines);

    if effective == 1 {
        for block_idx in 0..blocks {
            let line_start = block_idx * block_lines;
            let lines_in_block = (plane_len - line_start).min(block_lines);
            let used = lines_in_block * depth;
            with_cached_split_scratch(block_lines * depth, |scratch_real, scratch_imag| {
                let scratch_real = &mut scratch_real[..used];
                let scratch_imag = &mut scratch_imag[..used];

                for local_line in 0..lines_in_block {
                    let line = line_start + local_line;
                    let dst_real = &mut scratch_real[local_line * depth..(local_line + 1) * depth];
                    let dst_imag = &mut scratch_imag[local_line * depth..(local_line + 1) * depth];
                    for z in 0..depth {
                        let idx = z * plane_len + line;
                        dst_real[z] = real[idx];
                        dst_imag[z] = imag[idx];
                    }
                }

                with_cached_fft(depth, |fft| {
                    fft.execute_split_batch_in_place(
                        scratch_real, scratch_imag, lines_in_block, direction,
                    );
                });

                for local_line in 0..lines_in_block {
                    let line = line_start + local_line;
                    let src_real = &scratch_real[local_line * depth..(local_line + 1) * depth];
                    let src_imag = &scratch_imag[local_line * depth..(local_line + 1) * depth];
                    for z in 0..depth {
                        let idx = z * plane_len + line;
                        real[idx] = src_real[z];
                        imag[idx] = src_imag[z];
                    }
                }
            });
        }
        return;
    }

    let real_addr = real.as_mut_ptr() as usize;
    let imag_addr = imag.as_mut_ptr() as usize;
    (0..blocks).into_par_iter().for_each(|block_idx| {
        let real_ptr = real_addr as *mut f64;
        let imag_ptr = imag_addr as *mut f64;
        let line_start = block_idx * block_lines;
        let lines_in_block = (plane_len - line_start).min(block_lines);
        let used = lines_in_block * depth;

        with_cached_split_scratch(block_lines * depth, |scratch_real, scratch_imag| {
            let scratch_real = &mut scratch_real[..used];
            let scratch_imag = &mut scratch_imag[..used];

            for local_line in 0..lines_in_block {
                let line = line_start + local_line;
                let dst_real = &mut scratch_real[local_line * depth..(local_line + 1) * depth];
                let dst_imag = &mut scratch_imag[local_line * depth..(local_line + 1) * depth];
                for z in 0..depth {
                    let idx = z * plane_len + line;
                    unsafe {
                        dst_real[z] = *real_ptr.add(idx);
                        dst_imag[z] = *imag_ptr.add(idx);
                    }
                }
            }

            with_cached_fft(depth, |fft| {
                fft.execute_split_batch_in_place(
                    scratch_real, scratch_imag, lines_in_block, direction,
                );
            });

            for local_line in 0..lines_in_block {
                let line = line_start + local_line;
                let src_real = &scratch_real[local_line * depth..(local_line + 1) * depth];
                let src_imag = &scratch_imag[local_line * depth..(local_line + 1) * depth];
                for z in 0..depth {
                    let idx = z * plane_len + line;
                    unsafe {
                        *real_ptr.add(idx) = src_real[z];
                        *imag_ptr.add(idx) = src_imag[z];
                    }
                }
            }
        });
    });
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
    static TLS_2D_PLAN_CACHE: RefCell<HashMap<(usize, usize), FFT2Plan>> = RefCell::new(HashMap::new());
    static TLS_3D_PLAN_CACHE: RefCell<HashMap<(usize, usize, usize), FFT3Plan>> = RefCell::new(HashMap::new());
    static TLS_SPLIT_SCRATCH_CACHE: RefCell<HashMap<usize, (Vec<f64>, Vec<f64>)>> = RefCell::new(HashMap::new());
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
fn with_cached_fft2_plan<R>(width: usize, height: usize, f: impl FnOnce(&mut FFT2Plan) -> R) -> R {
    TLS_2D_PLAN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let plan = cache
            .entry((width, height))
            .or_insert_with(|| FFT2Plan::new(width, height));
        f(plan)
    })
}

#[cfg(feature = "parallel")]
fn with_cached_fft3_plan<R>(
    width: usize,
    height: usize,
    depth: usize,
    f: impl FnOnce(&mut FFT3Plan) -> R,
) -> R {
    TLS_3D_PLAN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let plan = cache
            .entry((width, height, depth))
            .or_insert_with(|| FFT3Plan::new(width, height, depth));
        f(plan)
    })
}

#[cfg(feature = "parallel")]
fn with_cached_split_scratch<R>(
    len: usize,
    f: impl FnOnce(&mut [f64], &mut [f64]) -> R,
) -> R {
    TLS_SPLIT_SCRATCH_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let (real, imag) = cache
            .entry(len)
            .or_insert_with(|| (vec![0.0; len], vec![0.0; len]));
        f(real.as_mut_slice(), imag.as_mut_slice())
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
    let target_tasks = effective.max(1);
    jobs.div_ceil(target_tasks).max(1)
}

#[cfg(feature = "parallel")]
fn process_contiguous_lines_impl(
    data: &mut [Complex<f64>],
    line_len: usize,
    jobs: usize,
    effective: usize,
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
}

#[cfg(feature = "parallel")]
fn process_contiguous_lines(
    data: &mut [Complex<f64>],
    line_len: usize,
    jobs: usize,
    workers: i32,
    direction: ffi::FFTDirection,
) {
    with_worker_pool(workers, jobs, |effective| {
        process_contiguous_lines_impl(data, line_len, jobs, effective, direction);
    });
}

#[cfg(feature = "parallel")]
fn process_split_lines_impl(
    real: &mut [f64],
    imag: &mut [f64],
    line_len: usize,
    jobs: usize,
    effective: usize,
    direction: ffi::FFTDirection,
) {
    assert!(line_len > 0, "line_len must be positive");
    assert!(jobs > 0, "jobs must be positive");
    let expected = line_len
        .checked_mul(jobs)
        .expect("line_len * jobs overflow in process_split_lines");
    assert_eq!(real.len(), expected, "real split buffer length mismatch");
    assert_eq!(imag.len(), expected, "imag split buffer length mismatch");

    if effective == 1 {
        with_cached_fft(line_len, |fft| {
            fft.execute_split_batch_in_place(real, imag, jobs, direction);
        });
        return;
    }

    let grouped_lines = lines_per_group(jobs, effective);
    let elems_per_group = line_len
        .checked_mul(grouped_lines)
        .expect("line_len * grouped_lines overflow in process_split_lines");

    real.par_chunks_mut(elems_per_group)
        .zip(imag.par_chunks_mut(elems_per_group))
        .for_each(|(real_group, imag_group)| {
            with_cached_fft(line_len, |fft| {
                let jobs_in_group = real_group.len() / line_len;
                fft.execute_split_batch_in_place(real_group, imag_group, jobs_in_group, direction);
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
    with_cached_fft2_plan(width, height, |plan| {
        plan.execute_with_workers(data, workers, ffi::kFFTDirection_Forward)
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
    with_cached_fft2_plan(width, height, |plan| {
        plan.execute_with_workers(data, workers, ffi::kFFTDirection_Inverse)
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
    let _ = wh;
    with_cached_fft3_plan(width, height, depth, |plan| {
        plan.execute_with_workers(data, workers, ffi::kFFTDirection_Forward)
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
    let _ = wh;
    with_cached_fft3_plan(width, height, depth, |plan| {
        plan.execute_with_workers(data, workers, ffi::kFFTDirection_Inverse)
    });
}

#[cfg(not(feature = "parallel"))]
fn execute_fft2_line_buffered(
    data: &mut [Complex<f64>],
    width: usize,
    height: usize,
    direction: ffi::FFTDirection,
) {
    let mut row_fft = AccelerateFFT::new(width);
    for row in data.chunks_exact_mut(width) {
        if direction == ffi::kFFTDirection_Forward {
            row_fft.forward(row);
        } else {
            row_fft.inverse(row);
        }
    }

    let mut col_fft = AccelerateFFT::new(height);
    let mut column = vec![Complex::new(0.0, 0.0); height];
    for x in 0..width {
        for y in 0..height {
            column[y] = data[y * width + x];
        }
        if direction == ffi::kFFTDirection_Forward {
            col_fft.forward(&mut column);
        } else {
            col_fft.inverse(&mut column);
        }
        for y in 0..height {
            data[y * width + x] = column[y];
        }
    }
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

    #[cfg(feature = "parallel")]
    {
        fft2_forward_with_workers_impl(data, width, height, 1);
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        execute_fft2_line_buffered(data, width, height, ffi::kFFTDirection_Forward);
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

    #[cfg(feature = "parallel")]
    {
        ifft2_with_workers_impl(data, width, height, 1);
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        execute_fft2_line_buffered(data, width, height, ffi::kFFTDirection_Inverse);
    }
}

#[cfg(not(feature = "parallel"))]
fn execute_fft3_line_buffered(
    data: &mut [Complex<f64>],
    width: usize,
    height: usize,
    depth: usize,
    direction: ffi::FFTDirection,
) {
    let wh = width
        .checked_mul(height)
        .expect("width * height overflow in execute_fft3_line_buffered");

    let mut x_fft = AccelerateFFT::new(width);
    for z in 0..depth {
        let z_base = z * wh;
        for y in 0..height {
            let start = z_base + y * width;
            let end = start + width;
            if direction == ffi::kFFTDirection_Forward {
                x_fft.forward(&mut data[start..end]);
            } else {
                x_fft.inverse(&mut data[start..end]);
            }
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
            if direction == ffi::kFFTDirection_Forward {
                y_fft.forward(&mut y_line);
            } else {
                y_fft.inverse(&mut y_line);
            }
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
            if direction == ffi::kFFTDirection_Forward {
                z_fft.forward(&mut z_line);
            } else {
                z_fft.inverse(&mut z_line);
            }
            for z in 0..depth {
                data[z * wh + y * width + x] = z_line[z];
            }
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

    #[cfg(feature = "parallel")]
    {
        fft3_with_workers_impl(data, width, height, depth, 1);
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        execute_fft3_line_buffered(data, width, height, depth, ffi::kFFTDirection_Forward);
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

    #[cfg(feature = "parallel")]
    {
        ifft3_with_workers_impl(data, width, height, depth, 1);
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        execute_fft3_line_buffered(data, width, height, depth, ffi::kFFTDirection_Inverse);
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
