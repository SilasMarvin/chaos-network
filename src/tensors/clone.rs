use crate::tensors::Tensor0D;
use std::clone::Clone;

impl Clone for Tensor0D {
    fn clone(&self) -> Self {
        Self::new_without_tape(self.data)
    }
}
