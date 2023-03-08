mod helpers;
mod new;
mod tensor;

pub use helpers::{element_wise_addition, element_wise_mul};
pub use tensor::{Tensor, Tensor0D, Tensor1D};
