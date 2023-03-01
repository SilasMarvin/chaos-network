use termion::input::TermRead;

use crate::network::{Network, NetworkMode};
use crate::tensors::{Tensor, Tensor1D};
use std::boxed::Box;

#[derive(Debug, PartialEq, Eq)]
enum TrainingPhase {
    Growing,
    Pruning,
}

#[derive(Debug, PartialEq, Eq)]
enum Action {
    Prune,
    Grow,
}

fn parse_input(inp: String) -> Result<(Action, f64, f64), Box<dyn std::error::Error>> {
    let split: Vec<&str> = inp.split(" ").collect();
    if split.len() > 3 {
        return Err(Box::<dyn std::error::Error>::from(
            "Length cannot be greater than 3",
        ));
    }
    let action = match split[0] {
        "grow" => Action::Grow,
        "prune" => Action::Prune,
        _ => return Err(Box::<dyn std::error::Error>::from("Action not found")),
    };
    if action == Action::Grow {
        if split.len() < 3 {
            return Err(Box::<dyn std::error::Error>::from(
                "The Growing Action requires 3 inputs",
            ));
        }
        let num1 = split[1].parse::<f64>()?;
        let num2 = split[2].parse::<f64>()?;
        Ok((Action::Grow, num1, num2))
    } else {
        Ok((Action::Prune, 0., 0.))
    }
}

pub trait StandardNetworkHandler<const N: usize> {
    fn train(&mut self);
    fn train_next_batch(&mut self);
    fn validate_next_batch(&mut self) -> f64;
    fn grow(&mut self, percent_nodes_to_add: f64, percent_connections_to_add: f64);
    fn prune(&mut self) -> bool;
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
        let mut stdin = termion::async_stdin();
        let mut pruning = false;
        for training_step in 0..self.max_training_steps {
            // Do training
            for _mini_step in 0..self.steps_per_training_step {
                self.train_next_batch();
            }

            // Do some evaluation
            let average_validation_accuracy: f64 = (0..self.validation_steps)
                .map(|_x| self.validate_next_batch())
                .sum::<f64>()
                / (self.validation_steps as f64);
            println!(
                "Step: {} | Pruning: {:?} | VAccuracy: {} | {:?}",
                training_step, pruning, average_validation_accuracy, self.network
            );
            self.past_validation_accuracy
                .push(average_validation_accuracy);

            // Perform phase action
            if pruning {
                self.prune();
            }

            // Match user input
            if let Some(line) = stdin.read_line().unwrap() {
                if line.len() > 0 {
                    let x = parse_input(line);
                    match x {
                        Ok((action, percent_nodes_to_add, percent_connections_to_add)) => {
                            match action {
                                Action::Grow => {
                                    self.grow(percent_nodes_to_add, percent_connections_to_add)
                                }
                                Action::Prune => pruning = !pruning,
                            }
                        }
                        Err(error) => println!("ERROR: {:?}", error),
                    }
                }
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

    fn grow(&mut self, percent_nodes_to_add: f64, percent_connections_to_add: f64) {
        let nodes_to_add =
            ((self.network.nodes.len() as f64 * (percent_nodes_to_add / 100.)) as usize).max(1);
        let connections_to_add = ((self.network.get_connection_count() as f64
            * (percent_connections_to_add / 100.)) as usize)
            .max(1);
        self.network
            .add_nodes(super::NodeKind::Normal, nodes_to_add);
        self.network.add_random_connections(connections_to_add);
        self.network.set_mode(NetworkMode::Training);
    }

    fn prune(&mut self) -> bool {
        let connections_to_remove = (self.network.get_connection_count() as f64 * 0.001) as usize;
        self.network
            .remove_weighted_connections(connections_to_remove);
        self.network.prune_unconnected_nodes();
        true
    }
}
