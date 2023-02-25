use crate::network::{Network, NetworkMode};
use crate::tensors::{Tensor, Tensor1D};
use std::boxed::Box;

const MORPHING_PERSPECTIVE_WINDOWS: usize = 3;
const EXPECTED_DIFF: f64 = 0.02;
const MAX_MORPH_CHANGE: f64 = 0.005;

pub trait StandardNetworkHandler<const N: usize> {
    fn train(&mut self);
    fn train_next_batch(&mut self);
    fn validate_next_batch(&mut self) -> f64;
    fn morph(&mut self);
}

pub struct StandardClassificationNetworkHandler<const N: usize> {
    network: Network<N>,
    max_training_steps: usize,
    steps_per_training_step: usize,
    mini_validation_steps: usize,
    train_data: Box<dyn Iterator<Item = (Vec<usize>, Vec<Tensor1D<N>>)>>,
    test_data: Box<dyn Iterator<Item = (Vec<usize>, Vec<Tensor1D<N>>)>>,
    past_validation_accuracy: Vec<f64>,
    morphing_perspective_window: usize,
    validation_steps: usize,
    validation_frequency: usize,
}

impl<const N: usize> StandardClassificationNetworkHandler<N> {
    pub fn new(
        network: Network<N>,
        max_training_steps: usize,
        steps_per_training_step: usize,
        mini_validation_steps: usize,
        train_data: Box<dyn Iterator<Item = (Vec<usize>, Vec<Tensor1D<N>>)>>,
        test_data: Box<dyn Iterator<Item = (Vec<usize>, Vec<Tensor1D<N>>)>>,
        morphing_perspective_window: usize,
        validation_steps: usize,
        validation_frequency: usize,
    ) -> Self {
        Self {
            network,
            max_training_steps,
            steps_per_training_step,
            mini_validation_steps,
            train_data,
            test_data,
            past_validation_accuracy: Vec::new(),
            morphing_perspective_window,
            validation_steps,
            validation_frequency,
        }
    }
}

impl<const N: usize> StandardNetworkHandler<N> for StandardClassificationNetworkHandler<N> {
    fn train(&mut self) {
        for training_step in 0..self.max_training_steps {
            // Do training
            for _mini_step in 0..self.steps_per_training_step {
                self.train_next_batch();
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
            self.past_validation_accuracy
                .push(average_validation_accuracy);

            // Maybe perform a more thorough validation of performance
            if training_step % self.validation_frequency == 0 {
                let average_validation_accuracy: f64 = (0..self.validation_steps)
                    .map(|_x| self.validate_next_batch())
                    .sum::<f64>()
                    / (self.validation_steps as f64);
                println!(
                    "\n{:?}\nThorough Validation Accuracy: {}",
                    self.network, average_validation_accuracy
                );
            }

            // Morph
            self.morph();
        }
    }

    fn train_next_batch(&mut self) {
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

    fn morph(&mut self) {
        if self.past_validation_accuracy.len()
            < self.morphing_perspective_window * MORPHING_PERSPECTIVE_WINDOWS
        {
            return;
        }

        if self.network.nodes.len() > 1500 {
            return;
        }

        // A TEST WILL DEFINITELY CHANGE
        let clustered: Vec<f64> =
            self.past_validation_accuracy[self.past_validation_accuracy.len()
                - self.morphing_perspective_window * MORPHING_PERSPECTIVE_WINDOWS..]
                .chunks(self.morphing_perspective_window)
                .map(|x| x.iter().sum())
                .collect();
        let clustered_windows = clustered[..clustered.len() - 1].windows(2);
        let clustered_windows_len = clustered_windows.len();
        let past_average_change =
            clustered_windows.map(|x| x[1] - x[0]).sum::<f64>() / (clustered_windows_len as f64);
        let current_change = clustered[clustered.len() - 1] - clustered[clustered.len() - 2];
        let current_change_diff = past_average_change - current_change;
        let morph_amount = (EXPECTED_DIFF / current_change_diff).min(1.);
        let nodes_to_add =
            (self.network.nodes.len() as f64 * MAX_MORPH_CHANGE * morph_amount) as usize;
        let connections_to_add =
            (self.network.get_connection_count() as f64 * MAX_MORPH_CHANGE * morph_amount) as usize;
        self.network
            .add_nodes(super::NodeKind::Normal, nodes_to_add);
        // self.network.add_random_connections(connections_to_add);
    }
}
