use crate::gradients::Tape;
use crate::tensors::{Tensor0D, Tensor1D};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::RwLock;

pub static TENSOR_COUNT: AtomicU64 = AtomicU64::new(0);

impl<const N: usize> Tensor0D<N> {
    pub fn new_with_tape(data: f64, tape: Option<Arc<RwLock<Tape<N>>>>) -> Self {
        let id_grad_for = TENSOR_COUNT.fetch_add(1, Ordering::SeqCst);
        Self {
            id: id_grad_for,
            grad_for: id_grad_for,
            data,
            tape,
        }
    }

    pub fn new_without_tape(data: f64) -> Self {
        let id_grad_for = TENSOR_COUNT.fetch_add(1, Ordering::SeqCst);
        Self {
            id: id_grad_for,
            grad_for: id_grad_for,
            data,
            tape: None,
        }
    }
}

impl<const N: usize> Tensor1D<N> {
    pub fn new_with_tape(data: [f64; N], tape: Option<Arc<RwLock<Tape<N>>>>) -> Self {
        // let id_grad_for = u64::MAX;
        let id_grad_for = TENSOR_COUNT.fetch_add(1, Ordering::SeqCst);
        Self {
            id: id_grad_for,
            grad_for: id_grad_for,
            data,
            tape,
        }
    }

    pub fn new_without_tape(data: [f64; N]) -> Self {
        // let id_grad_for = u64::MAX;
        let id_grad_for = TENSOR_COUNT.fetch_add(1, Ordering::SeqCst);
        Self {
            id: id_grad_for,
            grad_for: id_grad_for,
            data,
            tape: None,
        }
    }
}
