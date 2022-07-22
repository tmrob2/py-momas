#![allow(non_snake_case)]

use pyo3::prelude::*;
pub mod dfa;
pub mod agent;
pub mod mamdp_model;
pub mod mdp;
pub mod c_binding;
pub mod utils;
pub mod stapu_model;
use dfa::dfa::*;
use agent::agent::*;
use mamdp_model::{state_utils::*, model::*, test::*};
use mdp::{model::MultiObjectiveMDP, algorithms::multiobjective_scheduler_synthesis};
use std::hash::Hash;
use hashbrown::HashMap;
use c_binding::suite_sparse::*;
extern crate blis_src;
extern crate cblas_sys;
use cblas_sys::{cblas_dcopy, cblas_dgemv, cblas_dscal, cblas_ddot};
use std::mem;
use float_eq::float_eq;
use stapu_model::state_utils::*;


pub fn reverse_key_value_pairs<T, U>(map: &HashMap<T, U>) -> HashMap<U, T> 
where T: Clone + Hash, U: Clone + Hash + Eq {
    map.into_iter().fold(HashMap::new(), |mut acc, (a, b)| {
        acc.insert(b.clone(), a.clone());
        acc
    })
}

// --------------------------------------------------------------------------------
//                                LinAlg lib bindings 
// --------------------------------------------------------------------------------

pub struct COO {
    pub nzmax: i32,
    pub nr: i32,
    pub nc: i32,
    pub i: Vec<i32>,
    pub j: Vec<i32>,
    pub x: Vec<f64>,
    pub nz: i32,
}

pub struct MatrixAttr {
    pub m: Vec<f64>, // matrix
    pub nr: usize, // number of rows
    pub nc: usize // number of columns
}

pub struct SparseMatrixAttr {
    pub m: *mut cs_di, // matrix handle
    pub nr: usize, // number of rows
    pub nc: usize, // number of columns
    pub nnz: usize // the number of non-zero elements
}

pub fn create_sparse_matrix(m: i32, n: i32, rows: &[i32], cols: &[i32], x: &[f64])
                            -> *mut cs_di {
    unsafe {
        let T: *mut cs_di = cs_di_spalloc(m, n, x.len() as i32, 1, 1);
        for (k, elem) in x.iter().enumerate() {
            cs_di_entry(T, rows[k], cols[k], *elem);
        }
        return T
    }
}

pub fn convert_to_compressed(T: *mut cs_di) -> *mut cs_di {
    unsafe {
        cs_di_compress(T)
    }
}

pub fn print_matrix(A: *mut cs_di) {
    unsafe {
        cs_di_print(A, 0);
    }
}

pub fn transpose(A: *mut cs_di, nnz: i32) -> *mut cs_di {
    unsafe {
        cs_di_transpose(A, nnz)
    }
}

pub fn sp_mm_multiply_f64(A: *mut cs_di, B: *mut cs_di) -> *mut cs_di {
    unsafe {
        cs_di_multiply(A, B)
    }
}

pub fn sp_mv_multiply_f64(A: *mut cs_di, x: &[f64], y: &mut [f64]) -> i32 {
    unsafe {
        cs_di_gaxpy(A, x.as_ptr(), y.as_mut_ptr())
    }
}

pub fn sp_add(A: *mut cs_di, B: *mut cs_di, alpha: f64, beta: f64) -> *mut cs_di {
    // equation of the form alpha * A + beta * B
    unsafe {
        cs_di_add(A, B, alpha, beta)
    }
}

pub fn spfree(A: *mut cs_di) {
    unsafe {
        cs_di_spfree(A);
    }
}

pub fn spalloc(m: i32, n: i32, nzmax: i32, values: i32, t: i32) -> *mut cs_di {
    unsafe {
        cs_di_spalloc(m, n, nzmax, values, t)
    }
}

#[allow(non_snake_case)]
pub fn add_vecs(x: &[f64], y: &mut [f64], ns: i32, alpha: f64) {
    unsafe {
        cblas_sys::cblas_daxpy(ns, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }
}

pub fn copy(x: &[f64], y: &mut [f64], ns: i32) {
    unsafe {
        cblas_dcopy(ns, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }
}

pub fn dscal(x: &mut [f64], ns: i32, alpha: f64) {
    unsafe {
        cblas_dscal(ns, alpha, x.as_mut_ptr(), 1);
    }
}

fn blas_dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    unsafe {
        cblas_ddot(v1.len() as i32, v1.as_ptr(), 1, v2.as_ptr(), 1)
    }
}

fn blas_matrix_vector_mulf64(matrix: &[f64], v: &[f64], m: i32, n: i32, result: &mut [f64]) {
    unsafe {
        cblas_dgemv(
            cblas_sys::CBLAS_LAYOUT::CblasColMajor,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            m,
            n,
            1.0,
            matrix.as_ptr(),
            m,
            v.as_ptr(),
            1,
            1.0,
            result.as_mut_ptr(),
            1
        )
    }
}

/// Converts a Sparse struct representing a matrix into a C struct for CSS Sparse matrix
/// the C struct doesn't really exist, it is a mutable pointer reference to the Sparse struct
pub fn sparse_to_cs(sparse: &COO) -> *mut cs_di {
    let T = create_sparse_matrix(
        sparse.nr,
        sparse.nc,
        &sparse.i[..],
        &sparse.j[..],
        &sparse.x[..]
    );
    convert_to_compressed(T)
}

/// Get the maximum difference in the differences vector x_new - x used in
/// value iteration. 
fn max_eps(x: &[f64]) -> f64 {
    *x.iter()
        .max_by(|a, b| a.partial_cmp(&b)
        .expect("NaNs no expected in vector x"))
        .unwrap()
}

/// Used for updating the Q matrix of size number of actions x number of states
fn update_qmat(q: &mut [f64], v: &[f64], row: usize, nr: usize) -> Result<(), String> {
    for (ii, val) in v.iter().enumerate() {
        q[ii * nr + row] = *val;
    }
    Ok(())
}

/// Compute the argmax values in the Q table to determine which actions to select next
fn max_values(x: &mut [f64], q: &[f64], pi: &mut [f64], ns: usize, na: usize) {
    for ii in 0..ns {
        let (imax, max) = q[ii*na..(ii + 1)*na]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)|
                    a.partial_cmp(b).expect("NaNs no expected in vector x"))
                .unwrap();
            pi[ii] = imax as f64;
            x[ii] = *max;
    }
}

fn update_policy(eps: &[f64], thresh: &f64, pi: &mut [f64], pi_new: &[f64], ns: usize, policy_stable: &mut bool) {
    for ii in 0..ns {
        if eps[ii] > *thresh {
            pi[ii] = pi_new[ii];
            *policy_stable = false
        }
    }
}

// --------------------------------------------------------------------------------
//                          Section on number formatting 
// --------------------------------------------------------------------------------

#[derive(Hash, Eq, PartialEq)]
pub struct Mantissa((u64, i16, i8));

impl Mantissa {
    pub fn new(val: f64) -> Mantissa {
        Mantissa(integer_decode(val))
    }
}

pub fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = unsafe { mem::transmute(val) };
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52 ) & 0x7ff ) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff ) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

/// This method will adjust any values close to zero as zeroes, correcting LP rounding errors
pub fn val_or_zero_one(val: &f64) -> f64 {
    if float_eq!(*val, 0., abs <= 0.25 * f64::EPSILON) {
        0.
    } else if float_eq!(*val, 1., abs <= 0.25 * f64::EPSILON) {
        1.
    } else {
        *val
    }
}


// ---------------------------------------------------------------------------------
//                    Python Wrapper Extension for Rust MOTAP Lib
// ---------------------------------------------------------------------------------

/// Function to wrap and export the module to Python
#[pymodule]
pub fn rust_motap(py: Python, module: &PyModule) -> PyResult<()> {
    //module.add_function(wrap_pyfunction!(do_something_with_dfa, module)?)?;
    //module.add_function(wrap_pyfunction!(find_neighbour_states, module)?)?;
    module.add_function(wrap_pyfunction!(multiobjective_scheduler_synthesis, module)?)?;
    module.add_class::<MultiObjectiveMDP>()?;
    module.add_class::<Agent>()?;
    module.add_class::<DFA>()?;
    module.add_class::<Mission>()?;
    module.add_class::<Team>()?;
    mamdp(py, module)?;
    stapu(py, module)?;
    Ok(())
}

fn mamdp(py: Python, rust_motap: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "mamdp")?;
    child_module.add_function(wrap_pyfunction!(mamdp_model::state_utils::construct_initial_state, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(test_get_all_transitions, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(build_model, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(assert_done, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(convert_to_multobj_mdp, child_module)?)?;
    child_module.add_class::<MAMDP>()?;
    rust_motap.add_submodule(child_module)?;
    Ok(())
} 

fn stapu(py: Python, rust_motap: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "stapu")?;
    child_module.add_function(wrap_pyfunction!(stapu_model::state_utils::construct_initial_state, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(stapu_model::state_utils::get_available_actions, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(stapu_model::state_utils::transitions, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(stapu_model::model::build_model, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(stapu_model::model::convert_to_multobj_mdp, child_module)?)?;
    child_module.add_class::<TeamState>()?;
    rust_motap.add_submodule(child_module)?;
    Ok(())
}

