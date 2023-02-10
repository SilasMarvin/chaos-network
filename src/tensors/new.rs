use crate::gradients::Tape;
use crate::tensors::Tensor0D;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

pub static TENSOR_COUNT: AtomicU64 = AtomicU64::new(0);

impl Tensor0D {
    pub fn new_with_tape(data: f64) -> Self {
        Self {
            id: TENSOR_COUNT.fetch_add(1, Ordering::SeqCst),
            data,
            tape: Some(Tape::default()),
        }
    }

    pub fn new_without_tape(data: f64) -> Self {
        Self {
            id: TENSOR_COUNT.fetch_add(1, Ordering::SeqCst),
            data,
            tape: None,
        }
    }
}
