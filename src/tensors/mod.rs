// mod clone;
mod helpers;
mod new;
mod tensor;

pub use helpers::{element_wise_addition, element_wise_mul};
pub use new::TENSOR_COUNT;
pub use tensor::{Tensor, Tensor0D, Tensor1D};
