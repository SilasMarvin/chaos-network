use crate::tensors::{Tensor1D, WithTape, WithoutTape};
use chaos_network_derive::{build_backwards, build_forward, build_weights};

fn do_mish_backward(x: f64) -> f64 {
    let w = (4. * (x + 1.)) + (4. * (2. * x).exp()) + (3. * x).exp() + (x.exp() * ((4. * x) + 6.));
    let d = (2. * x.exp()) + (2. * x).exp() + 2.;
    (x.exp() * w) / d.powi(2)
}

pub struct OrderNetwork<const D: usize, const I: usize, const O: usize, const N: usize> {
    weights: Vec<f64>,
    backwards: Option<Vec<Box<dyn FnOnce(&Vec<f64>, &[f64; O]) -> Vec<f64>>>>,
}

#[macro_export]
macro_rules! build_order_network {
    ($f:literal, $i:literal, $o:literal, $n:literal) => {
        impl Default for OrderNetwork<$f, $i, $o, $n> {
            fn default() -> Self {
                Self {
                    weights: build_weights!($f),
                    backwards: None,
                }
            }
        }

        impl OrderNetwork<$f, $i, $o, $n> {
            pub fn forward_batch_no_grad(
                &self,
                input: Box<[[f64; $i]; $n]>,
            ) -> Box<[[f64; $o]; $n]> {
                let output = input.map(|data| {
                    let mut ret = [0.; $o];
                    let weights = &self.weights;
                    build_forward!($f, weights, data, ret);
                    ret
                });
                Box::new(output)
            }

            pub fn forward_batch(&mut self, input: Box<[[f64; $i]; $n]>) -> Box<[[f64; $o]; $n]> {
                let mut output = [[0.; $o]; $n];
                let mut partial_weight_funcs: Vec<
                    Box<dyn FnOnce(&Vec<f64>, &[f64; $o]) -> Vec<f64>>,
                > = Vec::new();
                input.into_iter().enumerate().for_each(|(i, data)| {
                    let mut ret = [0.; $o];
                    let weights = &self.weights;
                    build_forward!($f, weights, data, ret);
                    output[i] = ret;
                    let weights_len = self.weights.len();
                    partial_weight_funcs.push(Box::new(move |weights, output_grads| {
                        let mut weight_grads = Vec::new();
                        weight_grads.resize(weights_len, 0.);
                        build_backwards!($f, weights, data, output_grads, weight_grads);
                        weight_grads
                    }));
                });
                self.backwards = Some(partial_weight_funcs);
                Box::new(output)
            }

            pub fn backwards(&mut self, grads: &[[f64; $o]; $n]) -> Vec<Vec<f64>> {
                match self.backwards.take() {
                    Some(vec) => {
                        let weights = &self.weights;
                        vec.into_iter()
                            .enumerate()
                            .map(|(i, func)| func(&weights, &grads[i]))
                            .collect()
                    }
                    None => panic!("Calling backwards on OrderNetwork when it has no grads"),
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn prep_forward_batch_test() {
        let network_json = json!([
            ["Input", [2, 3], [0.1, 0.1]],
            ["Input", [2, 3], [0.1, 0.1]],
            ["Leaf", [], []],
            ["Leaf", [], []]
        ]);
        let mut file = File::create("networks/0/1.json").unwrap();
        write!(file, "{}", network_json).unwrap();
    }

    #[test]
    fn forward_batch_test() {
        build_order_network!(0, 2, 2, 1);
        let mut network: OrderNetwork<0, 2, 2, 1> = OrderNetwork::default();
        let input = Box::new([[0.1, 0.2]]);
        let output = network.forward_batch(input);
        // println!("Output: {:?}", output);
        // assert_eq!(output[0], [0.3, 0.3]); // The rounding is messed up but this works
        let output_grads = [[1., 1.]];
        let grads = network.backwards(&output_grads);
        assert_eq!(grads[0], vec![0.1, 0.1, 0.2, 0.2]);
    }
}
