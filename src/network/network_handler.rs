use crate::network::{Network, NetworkMode};
use crate::tensors::{Tensor, Tensor1D};
use std::boxed::Box;

const MORPHING_PERSPECTIVE_WINDOWS: usize = 25;
const EXPECTED_DIFF: f64 = 0.02;
const MAX_MORPH_CHANGE: f64 = 0.005;

pub trait StandardNetworkHandler<const N: usize> {
    fn train(&mut self);
    fn train_next_batch(&mut self);
    fn validate_next_batch(&mut self) -> f64;
    fn grow(&mut self);
    fn prune(&mut self);
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

#[derive(Debug)]
enum TrainingPhase {
    Growing,
    Pruning,
}

impl<const N: usize> StandardNetworkHandler<N> for StandardClassificationNetworkHandler<N> {
    fn train(&mut self) {
        let mut phase = TrainingPhase::Growing;
        let mut current_phase_steps = 0;
        for training_step in 0..self.max_training_steps {
            // Do training
            for _mini_step in 0..self.steps_per_training_step {
                self.train_next_batch();
            }

            // Perform phase action
            match &phase {
                TrainingPhase::Growing => {
                    self.grow();
                }
                TrainingPhase::Pruning => {
                    self.prune();
                }
            }

            // Rapidly evaluate performance
            let average_validation_accuracy: f64 = (0..self.mini_validation_steps)
                .map(|_x| self.validate_next_batch())
                .sum::<f64>()
                / (self.mini_validation_steps as f64);
            println!(
                "Training Step: {} | Phase: {:?} | Rapid Validation Accuracy: {}",
                training_step, phase, average_validation_accuracy
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

            // Update phase count
            current_phase_steps += 1;
            if current_phase_steps == 25 {
                phase = match phase {
                    TrainingPhase::Growing => TrainingPhase::Pruning,
                    _ => TrainingPhase::Growing,
                };
                current_phase_steps = 0;
            }
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

    fn grow(&mut self) {
        // if self.past_validation_accuracy.len()
        //     < self.morphing_perspective_window * MORPHING_PERSPECTIVE_WINDOWS
        // {
        //     return;
        // }
        //
        // if self.network.nodes.len() > 1500 {
        //     return;
        // }

        // A TEST WILL DEFINITELY CHANGE
        // let morph_amount =
        //     (EXPECTED_DIFF / self.get_current_validation_change_difference()).min(1.);
        // let nodes_to_add =
        //     (self.network.nodes.len() as f64 * MAX_MORPH_CHANGE * morph_amount) as usize;
        // let connections_to_add =
        //     (self.network.get_connection_count() as f64 * MAX_MORPH_CHANGE * morph_amount) as usize;
        // self.network
        //     .add_nodes(super::NodeKind::Normal, nodes_to_add);
        // self.network.add_random_connections(connections_to_add);
    }

    fn prune(&mut self) {
        if self.past_validation_accuracy.len()
            < self.morphing_perspective_window + MORPHING_PERSPECTIVE_WINDOWS
        {
            return;
        }

        let average_windowed_validation_accuracy: Vec<f64> =
            self.past_validation_accuracy[self.past_validation_accuracy.len()
                - (self.morphing_perspective_window + MORPHING_PERSPECTIVE_WINDOWS)..]
                .windows(self.morphing_perspective_window)
                .map(|x| x.iter().sum::<f64>() / (x.len() as f64))
                .collect();
        let average_validation_accuracy_over_last_windows = average_windowed_validation_accuracy
            [..average_windowed_validation_accuracy.len() - 1]
            .iter()
            .sum::<f64>()
            / (MORPHING_PERSPECTIVE_WINDOWS as f64);
        let current_average_validation_accuracy =
            *average_windowed_validation_accuracy.last().unwrap();
        if current_average_validation_accuracy < average_validation_accuracy_over_last_windows {
            return;
        }
        let connections_to_remove = (self.network.get_connection_count() as f64 * 0.005) as usize;
        self.network
            .remove_weighted_connections(connections_to_remove);
        self.network.prune_unconnected_nodes();
    }
}
