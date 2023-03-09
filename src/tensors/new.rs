use crate::gradients::Tape;
use crate::tensors::{Tensor0D, Tensor1D};
use std::sync::Arc;
use std::sync::RwLock;

use super::Tensor;

impl<const N: usize> Tensor0D<N> {
    pub fn new_with_tape(data: f64, tape: Option<Arc<RwLock<Tape<N>>>>) -> Self {
        let mut n = Self {
            id: 0,
            grad_for: 0,
            data,
            tape: None,
        };
        n.set_tape(tape);
        n
    }

    pub fn new_without_tape(data: f64) -> Self {
        Self {
            id: 0,
            grad_for: 0,
            data,
            tape: None,
        }
    }
}

impl<const N: usize> Tensor1D<N> {
    pub fn new_with_tape(data: [f64; N], tape: Option<Arc<RwLock<Tape<N>>>>) -> Self {
        let mut n = Self {
            id: 0,
            grad_for: 0,
            data,
            tape: None,
        };
        n.set_tape(tape);
        n
    }

    pub fn new_without_tape(data: [f64; N]) -> Self {
        Self {
            id: 0,
            grad_for: 0,
            data,
            tape: None,
        }
    }
}
