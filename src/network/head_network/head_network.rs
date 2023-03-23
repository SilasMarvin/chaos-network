use crate::tensors::{element_wise_mul, Tensor1D, WithTape, WithoutTape};

use rand::Rng;

pub struct HeadNetwork<const I: usize, const O: usize, const N: usize> {
    weights: [[f64; O]; I],
    backwards: Option<
        Box<
            dyn FnOnce(&[[f64; N]; O], &[[f64; O]; I]) -> (Box<[[f64; O]; I]>, Box<[[f64; N]; I]>)
                + Send
                + Sync,
        >,
    >,
}

impl<const I: usize, const O: usize, const N: usize> Clone for HeadNetwork<I, O, N> {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            backwards: None,
        }
    }
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
        Self {
            weights,
            backwards: None,
        }
    }
}

impl<const I: usize, const O: usize, const N: usize> HeadNetwork<I, O, N> {
    // NOTE: The grads can be done more efficiently for sure
    // Some explenation on what is happening here:
    // -- This represents a fully connected layer
    // -- The iniitial for loop is pretty simple except for our partial accumulation of the weight
    // grads. We can only partiall acumulate the weight grads, no the input grads, because the
    // input grads have to  be joined on a + and + with * is not associative where as the
    // weight_grads are only *
    // If this is confusing, which I am sure it is, try drawing out the test example below, and it
    // will be a lot more clear. I am sure there are better ways to do this, and unfortunately I
    // will probably have to revist this
    pub fn forward_batch(&mut self, input: Box<[Tensor1D<N, WithTape>; I]>) -> Box<[[f64; N]; O]> {
        let mut ret = [[0.; N]; O];

        let mut partial_weight_grads = [[[1.; N]; O]; I];
        for i in 0..O {
            for ii in 0..I {
                for iii in 0..N {
                    ret[i][iii] += self.weights[ii][i] * input[ii].data[iii];
                    partial_weight_grads[ii][i][iii] = input[ii].data[iii];
                }
            }
        }

        self.backwards = Some(Box::new(move |grads, weights| {
            let mut weight_grads = [[0.; O]; I];
            let mut input_grads = [[0.; N]; I];
            for i in 0..I {
                for ii in 0..O {
                    // Get final weight grads
                    let g = element_wise_mul(&partial_weight_grads[i][ii], &grads[ii]);
                    let g: f64 = g.iter().sum();
                    weight_grads[i][ii] = g / (N as f64);
                    for iii in 0..N {
                        input_grads[i][iii] += weights[i][ii] * grads[ii][iii];
                    }
                }
            }
            (Box::new(weight_grads), Box::new(input_grads))
        }));

        Box::new(ret)
    }

    pub fn forward_batch_no_grad(&self, input: Box<[Tensor1D<N>; I]>) -> Box<[[f64; N]; O]> {
        let mut ret = [[0.; N]; O];
        for i in 0..O {
            for ii in 0..I {
                for iii in 0..N {
                    ret[i][iii] += self.weights[ii][i] * input[ii].data[iii];
                }
            }
        }
        Box::new(ret)
    }

    pub fn nll(t: Box<[[f64; N]; O]>, indexes: &[usize; N]) -> (Box<[f64; N]>, Box<[[f64; N]; O]>) {
        let sum_e = t.iter().fold([0.; N], |mut acc, data| {
            data.iter().enumerate().for_each(|(i, x)| acc[i] += x.exp());
            acc
        });
        let log_softmax: Vec<[f64; N]> = t
            .iter()
            .map(|data| {
                let mut tracker = 0;
                data.map(|x| {
                    let x = (x.exp() / sum_e[tracker]).ln();
                    tracker += 1;
                    x
                })
            })
            .collect();
        let losses: [f64; N] = indexes
            .iter()
            .enumerate()
            .map(|(ii, i)| -1. * log_softmax[*i][ii])
            .collect::<Vec<f64>>()
            .try_into()
            .unwrap();
        let grads: [[f64; N]; O] = t
            .into_iter()
            .enumerate()
            .map(|(i, data)| {
                let mut tracker = 0;
                data.map(|x| {
                    let mut x = x.exp() / sum_e[tracker];
                    if i == indexes[tracker] {
                        x -= 1.;
                    }
                    tracker += 1;
                    x
                })
            })
            .collect::<Vec<[f64; N]>>()
            .try_into()
            .unwrap();
        (Box::new(losses), Box::new(grads))
    }

    pub fn backwards(&mut self, grads: &[[f64; N]; O]) -> (Box<[[f64; O]; I]>, Box<[[f64; N]; I]>) {
        match self.backwards.take() {
            Some(func) => func(grads, &self.weights),
            None => panic!("Calling backwards on a HeadNetwork when it has no grads"),
        }
    }

    pub fn apply_gradients(&mut self, grads: &[[f64; O]; I]) {
        for i in 0..I {
            for ii in 0..O {
                self.weights[i][ii] -= 0.1 * grads[i][ii];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_batch() {
        // let mut network: HeadNetwork<2, 2, 1> = HeadNetwork {
        //     weights: [[0.1, 0.2], [0.3, 0.4]],
        //     backwards: None,
        // };
        // let input: Box<[Tensor1D<1, WithTape>; 2]> =
        //     Box::new([Tensor1D::new([0.1]), Tensor1D::new([0.2])]);
        // let indices = [0];
        // let outputs = network.forward_batch(input);
        // let (loss, nll_grads) = HeadNetwork::<2, 2, 1>::nll(outputs, &indices);
        // println!("Loss: {:?}", loss);
        // println!("NLL Grads: {:?}", nll_grads);
        // let (weight_grads, input_grads) = network.backwards.unwrap()(&nll_grads, &network.weights);
        // println!("Weight Grads: {:?}", weight_grads);
        // println!("Input Grads: {:?}", input_grads);

        let mut network: HeadNetwork<2, 2, 2> = HeadNetwork {
            weights: [[0.1, 0.2], [0.3, 0.4]],
            backwards: None,
        };
        let input: Box<[Tensor1D<2, WithTape>; 2]> =
            Box::new([Tensor1D::new([0.1, 0.3]), Tensor1D::new([0.2, 0.4])]);
        let indices = [0, 1];
        let outputs = network.forward_batch(input);
        println!("Outputs: {:?}", outputs);
        let (loss, nll_grads) = HeadNetwork::<2, 2, 2>::nll(outputs, &indices);
        println!("Loss: {:?}", loss);
        let (weight_grads, input_grads) = network.backwards.unwrap()(&nll_grads, &network.weights);
        println!("Weight Grads: {:?}", weight_grads);
        println!("Input Grads: {:?}", input_grads);
    }
}
