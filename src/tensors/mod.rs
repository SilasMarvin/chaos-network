mod helpers;
mod tensor;

pub use helpers::{element_wise_addition, element_wise_mul};
pub use tensor::{Tensor0D, Tensor1D, WithTape, WithoutTape};
