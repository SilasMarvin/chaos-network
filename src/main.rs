use rand::distributions::Uniform;
use rand::prelude::*;
use rust_mnist::Mnist;

mod gradients;
mod network;
mod tensor_operations;
mod tensors;

use crate::network::{Network, StandardClassificationNetworkHandler, StandardNetworkHandler};
use crate::tensors::Tensor1D;

const INPUTS: usize = 784;
const OUTPUTS: usize = 10;
const ADDITIONAL_STARTING_NODES: usize = 0;
const BATCH_SIZE: usize = 32;
const MAX_TRAINING_STEPS: usize = 1000000;
const STEPS_PER_TRAINING_STEPS: usize = 200;
const MINI_VALIDATION_STEPS: usize = 100;
const MORPHING_PERSPECTIVE_WINDOW: usize = 5;
const VALIDATION_STEPS: usize = 50;
const VALIDATION_FREQUENCY: usize = 1;

struct RepeatingNetworkData<const N: usize> {
    labels: Vec<u8>,
    data: Vec<[u8; 784]>,
    distribution: Uniform<usize>,
    rng: ThreadRng,
}

impl<const N: usize> RepeatingNetworkData<N> {
    fn new(labels: Vec<u8>, data: Vec<[u8; 784]>) -> Self {
        let distribution = Uniform::from(0..data.len());
        let rng = rand::thread_rng();
        RepeatingNetworkData {
            labels,
            data,
            distribution,
            rng,
        }
    }
}

impl<const N: usize> Iterator for RepeatingNetworkData<N> {
    type Item = (Vec<usize>, Vec<Tensor1D<N>>);
    fn next(&mut self) -> Option<Self::Item> {
        let indexes: Vec<usize> = (0..N)
            .map(|_i| self.distribution.sample(&mut self.rng))
            .collect();
        let inputs: Vec<Tensor1D<N>> = (0..INPUTS)
            .map(|i| {
                let data: [f64; N] = indexes
                    .iter()
                    .map(|ii| self.data[*ii][i] as f64 / 255.)
                    .collect::<Vec<f64>>()
                    .try_into()
                    .unwrap();
                Tensor1D::new_without_tape(data)
            })
            .collect();
        let labels: Vec<usize> = indexes
            .into_iter()
            .map(|i| self.labels[i] as usize)
            .collect();
        Some((labels, inputs))
    }
}

fn main() {
    // Load data
    let mnist = Mnist::new("data/");
    let train_data: RepeatingNetworkData<BATCH_SIZE> =
        RepeatingNetworkData::new(mnist.train_labels, mnist.train_data);
    let test_data: RepeatingNetworkData<BATCH_SIZE> =
        RepeatingNetworkData::new(mnist.test_labels, mnist.test_data);

    // Build the network
    let network = Network::new(INPUTS, ADDITIONAL_STARTING_NODES, OUTPUTS);

    // Build the network handler
    let mut network_handler = StandardClassificationNetworkHandler::new(
        network,
        MAX_TRAINING_STEPS,
        STEPS_PER_TRAINING_STEPS,
        MINI_VALIDATION_STEPS,
        Box::new(train_data),
        Box::new(test_data),
        MORPHING_PERSPECTIVE_WINDOW,
        VALIDATION_STEPS,
        VALIDATION_FREQUENCY,
    );

    // Train
    network_handler.train();
}
