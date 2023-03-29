use crate::tensors::{Tensor1D, WithTape, WithoutTape};

pub struct OrderNetwork<const D: usize, const I: usize, const O: usize, const N: usize> {
    pub weights: Vec<f64>,
    pub backwards: Option<Vec<Box<dyn FnOnce(&Vec<f64>, &[f64; O]) -> Vec<f64>>>>,
}

#[macro_export]
macro_rules! build_order_network {
    ($f:literal, $i:literal, $o:literal, $n:literal) => {
        use crate::network::OrderNetwork;
        use chaos_network_derive::{build_backwards, build_forward, build_weights};

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
    fn prep_forward_batch_test_1() {
        let network_json = json!([
            ["Input", [2, 3], [0.1, 0.2]],
            ["Input", [2, 3], [0.3, 0.4]],
            ["Leaf", [], [0.5]],
            ["Leaf", [], [0.6]]
        ]);
        let mut file = File::create("networks/0/1.json").unwrap();
        write!(file, "{}", network_json).unwrap();
    }

    #[test]
    // NOTE compare output to results of test_forward_batch_1 in python-tests tests/order_network.py
    fn forward_batch_test_1() {
        build_order_network!(0, 2, 2, 2);
        let mut network: OrderNetwork<0, 2, 2, 2> = OrderNetwork::default();
        let input = Box::new([[0.1, 0.2], [0.2, 0.3]]);
        let output = network.forward_batch(input);
        assert_eq!(output[0], [0.4384104584592529, 0.5611483776438518]); // The rounding is messed up but this works
        let output_grads = [[0.13, 0.14], [0.14, 0.15]];
        let grads = network.backwards(&output_grads);
        assert_eq!(
            grads[0],
            vec![
                0.01193199182293912,
                0.013563128154802607,
                0.02386398364587824,
                0.027126256309605214,
                0.1193199182293912,
                0.13563128154802606
            ]
        );
        assert_eq!(
            grads[1],
            vec![
                0.02616781305329546,
                0.02966924163153822,
                0.03925171957994319,
                0.04450386244730733,
                0.1308390652664773,
                0.1483462081576911
            ]
        );
    }

    #[test]
    fn prep_forward_batch_test_2() {
        let network_json = json!([
            ["Input", [2, 3, 4], [0.1, 0.2, 0.3]],
            ["Input", [2, 3, 5], [0.4, 0.5, 0.6]],
            ["Normal", [4, 5], [0.7, 0.8, 0.9]],
            ["Normal", [4, 5], [0.1, 0.2, 0.3]],
            ["Leaf", [], [0.4]],
            ["Leaf", [], [0.5]]
        ]);
        let mut file = File::create("networks/1/1.json").unwrap();
        write!(file, "{}", network_json).unwrap();
    }

    #[test]
    // NOTE compare output to results of test_forward_batch_2 in python-tests tests/order_network.py
    fn forward_batch_test_2() {
        build_order_network!(1, 2, 2, 1);
        let mut network: OrderNetwork<1, 2, 2, 1> = OrderNetwork::default();
        let input = Box::new([[0.1, 0.2]]);
        let output = network.forward_batch(input);
        assert_eq!(output[0], [0.8433231723714711, 1.130676581160498]); // The rounding is messed up but this works
        let output_grads = [[0.13, 0.14]];
        let grads = network.backwards(&output_grads);
        assert_eq!(
            grads[0],
            vec![
                0.024439393766277528,
                0.005343493033151527,
                0.013586083416021974,
                0.048878787532555056,
                0.010686986066303054,
                0.030250713786655427,
                0.24439393766277526,
                0.08826906549591601,
                0.09826975717606541,
                0.053434930331515265,
                0.020000194963494835,
                0.022266173222685117,
                0.13586083416021974,
                0.15125356893327713,
            ]
        );
    }

    #[test]
    fn prep_forward_batch_test_3() {
        let network_json = json!([
            ["Input", [2, 3, 5], [0.1, 0.2, 0.3]],
            ["Input", [2, 3, 6], [0.4, 0.5, 0.6]],
            ["Normal", [5, 6, 4], [0.7, 0.8, 0.9, 0.1]],
            ["Normal", [5, 6, 4], [0.2, 0.3, 0.4, 0.5]],
            ["Normal", [5, 6], [0.6, 0.7, 0.8]],
            ["Leaf", [], [0.9]],
            ["Leaf", [], [0.1]]
        ]);
        let mut file = File::create("networks/2/1.json").unwrap();
        write!(file, "{}", network_json).unwrap();
    }

    #[test]
    // NOTE compare output to results of test_forward_batch_3 in python-tests tests/order_network.py
    fn forward_batch_test_3() {
        build_order_network!(2, 2, 2, 1);
        let mut network: OrderNetwork<2, 2, 2, 1> = OrderNetwork::default();
        let input = Box::new([[0.1, 0.2]]);
        let output = network.forward_batch(input);
        assert_eq!(output[0], [1.9038374515943743, 1.2983920430328038]); // The rounding is messed up but this works
        let output_grads = [[0.13, 0.14]];
        let grads = network.backwards(&output_grads);
        assert_eq!(
            grads[0],
            vec![
                0.02698030983733234,
                0.016810568409272402,
                0.013929013687361748,
                0.05396061967466468,
                0.033621136818544804,
                0.030451715204576336,
                0.2698030983733234,
                0.09049709057529431,
                0.09892271236484003,
                0.141668736099644,
                0.16810568409272403,
                0.031168961562968844,
                0.03407090990222471,
                0.048793473492833556,
                0.21805184582827822,
                0.08867297231388396,
                0.09692876178647514,
                0.13929013687361746,
                0.15225857602288168
            ]
        );
    }
}
