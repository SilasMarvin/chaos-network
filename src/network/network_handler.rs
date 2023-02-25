use crate::network::{Network, NetworkMode};
use crate::tensors::{Tensor, Tensor1D};
use std::boxed::Box;

pub trait StandardNetworkHandler<const N: usize> {
    fn train(&mut self);
    fn train_batch(&mut self);
    fn validate_next_batch(&mut self) -> f64;
}

pub struct StandardClassificationNetworkHandler<const N: usize> {
    network: Network<N>,
    max_training_steps: usize,
    steps_per_training_step: usize,
    mini_validation_steps: usize,
    train_data: Box<dyn Iterator<Item = (Vec<usize>, Vec<Tensor1D<N>>)>>,
    test_data: Box<dyn Iterator<Item = (Vec<usize>, Vec<Tensor1D<N>>)>>,
}

impl<const N: usize> StandardClassificationNetworkHandler<N> {
    pub fn new(
        network: Network<N>,
        max_training_steps: usize,
        steps_per_training_step: usize,
        mini_validation_steps: usize,
        train_data: Box<dyn Iterator<Item = (Vec<usize>, Vec<Tensor1D<N>>)>>,
        test_data: Box<dyn Iterator<Item = (Vec<usize>, Vec<Tensor1D<N>>)>>,
    ) -> Self {
        Self {
            network,
            max_training_steps,
            steps_per_training_step,
            mini_validation_steps,
            train_data,
            test_data,
        }
    }
}

impl<const N: usize> StandardNetworkHandler<N> for StandardClassificationNetworkHandler<N> {
    fn train(&mut self) {
        for training_step in 0..self.max_training_steps {
            // Do training
            for _mini_step in 0..self.steps_per_training_step {
                self.train_batch();
            }

            // Rapidly evaluate performance
            let average_validation_accuracy: f64 = (0..self.mini_validation_steps)
                .map(|_x| self.validate_next_batch())
                .sum::<f64>()
                / (self.mini_validation_steps as f64);
            println!(
                "Training Step: {} -> Rapid Validation Accuracy: {}",
                training_step, average_validation_accuracy
            );
        }
    }

    fn train_batch(&mut self) {
        if self.network.mode != NetworkMode::Training {
            self.network.set_mode(NetworkMode::Training);
        }
        let (labels, inputs) = self.train_data.next().unwrap();
        let outputs = self.network.forward_batch(inputs);
        let loss = &mut Tensor1D::nll(outputs, labels);
        let grads = loss.backward();
        self.network.apply_gradients(grads);
    }

    fn validate_next_batch(&mut self) -> f64 {
        self.network.set_mode(NetworkMode::Inference);
        let (labels, inputs) = self.test_data.next().unwrap();
        let outputs = self.network.forward_batch(inputs);
        let guesses: Vec<usize> = (0..N)
            .map(|i| {
                let mut max: (usize, f64) = (0, outputs[0].data[i]);
                for ii in 0..self.network.leaves_count {
                    if outputs[ii].data[i] > max.1 {
                        max = (ii, outputs[ii].data[i]);
                    }
                }
                max.0
            })
            .collect();
        let correct = guesses
            .iter()
            .enumerate()
            .filter(|(i, g)| **g == labels[*i])
            .count();
        (correct as f64) / N as f64
    }
}
