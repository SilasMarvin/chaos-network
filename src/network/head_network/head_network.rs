use crate::tensors::{element_wise_mul, Tensor1D, WithTape, WithoutTape};

use matrixmultiply::dgemm;
use rand::Rng;
use std::fs::File;
use std::io::Write;

pub struct HeadNetwork<const I: usize, const O: usize, const N: usize> {
    pub weights: [[f64; O]; I],
    backwards: Option<
        Box<
            dyn FnOnce(&[[f64; O]; N], &[[f64; O]; I]) -> (Box<[[f64; O]; I]>, Box<[[f64; I]; N]>)
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
    pub fn forward_batch(&mut self, input: Box<[[f64; I]; N]>) -> Box<[[f64; O]; N]> {
        let mut ret = Box::new([[0.; O]; N]);
        unsafe {
            dgemm(
                N,
                I,
                O,
                1.,
                input[0].as_ptr(),
                I as isize,
                1,
                self.weights[0].as_ptr(),
                O as isize,
                1,
                0.,
                ret[0].as_mut_ptr(),
                O as isize,
                1,
            );
        }

        self.backwards = Some(Box::new(move |grads, weights| {
            let mut weight_grads = [[0.; O]; I];
            let mut input_grads = [[0.; I]; N];
            for i in 0..N {
                for ii in 0..I {
                    let mut tracker = 0;
                    weight_grads[ii] = grads[i].map(|d| {
                        let x = d * input[i][ii];
                        let x = x / (N as f64);
                        let x = x + weight_grads[ii][tracker];
                        tracker += 1;
                        x
                    });
                    // I thought dividing by N only made sense at the end, but it seems pytorch
                    // wants to do it here as well
                    input_grads[i][ii] = element_wise_mul(&weights[ii], &grads[i])
                        .iter()
                        .sum::<f64>()
                        / (N as f64);
                }
            }
            (Box::new(weight_grads), Box::new(input_grads))
        }));

        ret
    }

    pub fn forward_batch_no_grad(&mut self, input: Box<[[f64; I]; N]>) -> Box<[[f64; O]; N]> {
        let mut ret = Box::new([[0.; O]; N]);
        unsafe {
            dgemm(
                N,
                I,
                O,
                1.,
                input[0].as_ptr(),
                I as isize,
                1,
                self.weights[0].as_ptr(),
                O as isize,
                1,
                0.,
                ret[0].as_mut_ptr(),
                O as isize,
                1,
            );
        }
        ret
    }

    pub fn nll(t: Box<[[f64; O]; N]>, indices: &[usize; N]) -> (Box<[f64; N]>, Box<[[f64; O]; N]>) {
        let sum_e: [f64; N] = t.map(|o| o.iter().map(|d| d.exp()).sum());
        let mut tracker = 0;
        let log_softmax: [[f64; O]; N] = t.map(|o| {
            let x = o.map(|d| (d.exp() / sum_e[tracker]).ln());
            tracker += 1;
            x
        });
        let mut tracker = 0;
        let loss = indices.map(|i| {
            let x = -1. * log_softmax[tracker][i];
            tracker += 1;
            x
        });
        let mut tracker = 0;
        let grads = t.map(|o| {
            let mut inner_tracker = 0;
            let x = o.map(|d| {
                let mut g = d.exp() / sum_e[tracker];
                if inner_tracker == indices[tracker] {
                    g -= 1.;
                }
                inner_tracker += 1;
                g
            });
            tracker += 1;
            x
        });
        (Box::new(loss), Box::new(grads))
    }

    pub fn backwards(&mut self, grads: &[[f64; O]; N]) -> (Box<[[f64; O]; I]>, Box<[[f64; I]; N]>) {
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

    pub fn write_to_dir(&self, path: &str) -> std::io::Result<()> {
        // Serde does not serialize big arrays
        let write_out: Vec<Vec<f64>> = self
            .weights
            .clone()
            .into_iter()
            .map(|x| x.to_vec())
            .collect();
        let write_out = serde_json::to_string(&write_out)?;
        let path: String = format!("{}/head-network.json", path);
        let mut file = File::create(path)?;
        write!(file, "{}", write_out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[test]
    fn test_head_network_forward_batch() {
        let mut network: HeadNetwork<2, 2, 2> = HeadNetwork {
            weights: [[0.1, 0.2], [0.3, 0.4]],
            backwards: None,
        };
        let input = Box::new([[0.1, 0.2], [0.3, 0.4]]);
        let outputs = network.forward_batch(input);
        assert_eq!(
            *outputs,
            [[0.07, 0.10000000000000002], [0.15, 0.22000000000000003]]
        );
        let indices = [0, 1];
        let (loss, nll_grads) = HeadNetwork::<2, 2, 2>::nll(outputs, &indices);
        assert_eq!(*loss, [0.7082596763414484, 0.658759555548697]);
        assert_eq!(
            *nll_grads,
            [
                [-0.5074994375506203, 0.5074994375506204],
                [0.48250714233361025, -0.48250714233361025]
            ]
        );
        let (weight_grads, input_grads) = network.backwards(&nll_grads);
        assert_eq!(
            *weight_grads,
            [
                [0.04700109947251052, -0.047001099472510514],
                [0.04575148471166002, -0.045751484711660004]
            ]
        );
        assert_eq!(
            *input_grads,
            [
                [0.02537497187753103, 0.025374971877531044],
                [-0.024125357116680513, -0.024125357116680513]
            ]
        );
    }

    // #[test]
    // fn test_head_network_forward_batch_train() {
    //     let mut network: HeadNetwork<2, 2, 2> = HeadNetwork {
    //         weights: [[0.1, 0.2], [0.3, 0.4]],
    //         backwards: None,
    //         test_backwards: None,
    //     };
    //     let input = Box::new([[0.1, 0.2], [0.3, 0.4]]);
    //     for i in 0..100 {
    //         let outputs = network.forward_batch(input.clone());
    //         let indices = [1, 1];
    //         let (loss, nll_grads) = HeadNetwork::<2, 2, 2>::nll_test(outputs, &indices);
    //         println!("Loss: {:?}", loss);
    //         let (weight_grads, input_grads) = network.backwards_test(&nll_grads);
    //         network.apply_gradients(&weight_grads);
    //     }
    // }

    // #[test]
    // fn test_head_network_speed() {
    //     let mut network: HeadNetwork<1000, 10, 32> = HeadNetwork {
    //         weights: [[0.1; 10]; 1000],
    //         backwards: None,
    //     };
    //
    //     // let now = std::time::Instant::now();
    //     // let indices = [0; 32];
    //     // for _i in 0..1000 {
    //     //     let input: [Tensor1D<32, WithTape>; 1000] = (0..1000)
    //     //         .into_iter()
    //     //         .map(|_i| Tensor1D::new([0.; 32]))
    //     //         .collect::<Vec<Tensor1D<32, WithTape>>>()
    //     //         .try_into()
    //     //         .unwrap();
    //     //     let input = Box::new(input);
    //     //     let outputs = network.forward_batch(input);
    //     //     let (_, nll_grads) = HeadNetwork::<1000, 10, 32>::nll(outputs, &indices);
    //     //     let (head_network_grads, _) = network.backwards(&nll_grads);
    //     //     network.apply_gradients(&head_network_grads);
    //     // }
    //     // let elapsed_time = now.elapsed();
    //     // println!(
    //     //     "Head Network Elapsed Time With Backwards: {:?}",
    //     //     elapsed_time.as_secs()
    //     // );
    //
    //     let now = std::time::Instant::now();
    //     for _i in 0..1000 {
    //         let input: [Tensor1D<32>; 1000] = (0..1000)
    //             .into_iter()
    //             .map(|_i| Tensor1D::new([0.; 32]))
    //             .collect::<Vec<Tensor1D<32>>>()
    //             .try_into()
    //             .unwrap();
    //         let input = Box::new(input);
    //         let _outputs = network.forward_batch_no_grad(input);
    //     }
    //     let elapsed_time = now.elapsed();
    //     println!(
    //         "Head Network Elapsed Time Without Backwards: {:?}",
    //         elapsed_time.as_secs()
    //     );
    // }
}
