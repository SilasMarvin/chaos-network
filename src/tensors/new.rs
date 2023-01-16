use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;
use crate::gradients::Tape;
use crate::tensors::Tensor0D;


pub static TENSOR_COUNT: AtomicI32 = AtomicI32::new(0);


impl Tensor0D {
    pub fn new_with_tape(data: f32) -> Self {
        Self {
            id: TENSOR_COUNT.fetch_add(1, Ordering::SeqCst),
            data,
            tape: Some(Tape::new())
        }
    }

    pub fn new_without_tape(data: f32) -> Self {
        Self {
            id: TENSOR_COUNT.fetch_add(1, Ordering::SeqCst),
            data,
            tape: None 
        }
    }
}
