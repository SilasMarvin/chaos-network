use crate::tensors::{Tensor1D, WithTape, WithoutTape};
use chaos_network_derive::{build_forward, build_weights};

pub struct OrderNetwork<const L: usize, const N: usize> {
    weights: Vec<f64>,
}

impl<const L: usize, const N: usize> Default for OrderNetwork<L, N> {
    fn default() -> Self {
        Self {
            weights: build_weights!(),
        }
    }
}

impl<const L: usize, const N: usize> OrderNetwork<L, N> {
    pub fn forward_batch_no_grad(&self) {
        let mut input: Vec<f64> = Vec::new();
        input.resize(1000, 2.);
        let output = self.forward_no_grad(&input);
        println!("{:?}", output);
    }

    fn forward_no_grad(&self, input: &Vec<f64>) -> Vec<f64> {
        let mut ret: Vec<f64> = Vec::new();
        ret.resize(10, 0.);

        build_forward!(self, input, ret);
        ret
    }
}
