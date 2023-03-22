use crate::tensors::{Tensor1D, WithTape, WithoutTape};

use rand::Rng;

pub struct HeadNetwork<const I: usize, const O: usize, const N: usize> {
    weights: [[f64; O]; I],
}

impl<const I: usize, const O: usize, const N: usize> Default for HeadNetwork<I, O, N> {
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = [[0.; O]; I];
        for i in 0..I {
            for ii in 0..O {
                let w = (rng.gen::<f64>() - 0.5) / 100.;
                weights[i][ii] = w;
            }
        }
        Self { weights }
    }
}

impl<const I: usize, const O: usize, const N: usize> HeadNetwork<I, O, N> {
    pub fn forward_batch(&mut self, input: Vec<Tensor1D<N, WithTape>>) {}
}
