use rust_mnist::Mnist;

mod gradients;
mod network;
mod tensor_operations;
mod tensors;

use crate::network::{Network, StandardClassificationNetworkHandler, StandardNetworkHandler};
use crate::tensors::Tensor1D;

const INPUTS: usize = 784;
const OUTPUTS: usize = 10;
const ADDITIONAL_STARTING_NODES: usize = 100;
const BATCH_SIZE: usize = 64;
const MAX_TRAINING_STEPS: usize = 1000000;
const STEPS_PER_TRAINING_STEPS: usize = 100;
const MINI_VALIDATION_STEPS: usize = 50;
const MORPHING_PERSPECTIVE_WINDOW: usize = 30;
const VALIDATION_STEPS: usize = 1000;
const VALIDATION_FREQUENCY: usize = 50;

struct RepeatingNetworkData<const N: usize> {
    current_index: usize,
    labels: Vec<u8>,
    data: Vec<[u8; 784]>,
}

impl<const N: usize> RepeatingNetworkData<N> {
    fn new(labels: Vec<u8>, data: Vec<[u8; 784]>) -> Self {
        RepeatingNetworkData {
            current_index: 0,
            labels,
            data,
        }
    }
}

impl<const N: usize> Iterator for RepeatingNetworkData<N> {
    type Item = (Vec<usize>, Vec<Tensor1D<N>>);
    fn next(&mut self) -> Option<Self::Item> {
        let start = self.current_index;
        let inputs: Vec<Tensor1D<N>> = (0..INPUTS)
            .map(|i| {
                let data: [f64; N] = (start..start + N)
                    .map(|ii| self.data[ii][i] as f64 / 255.)
                    .collect::<Vec<f64>>()
                    .try_into()
                    .unwrap();
                Tensor1D::new_without_tape(data)
            })
            .collect();
        let labels: Vec<usize> = (start..start + N)
            .map(|i| self.labels[i] as usize)
            .collect();
        self.current_index += N;
        if self.current_index + N > self.data.len() {
            self.current_index = 0;
        }
        Some((labels, inputs))
    }
}

fn build_network<const N: usize>() -> Network<N> {
    let mut network: Network<N> = Network::default();
    network.add_nodes(network::NodeKind::Leaf, OUTPUTS);
    network.add_nodes(network::NodeKind::Input, INPUTS);
    network.add_nodes(network::NodeKind::Normal, ADDITIONAL_STARTING_NODES);
    network
}

fn main() {
    // Load data
    let mnist = Mnist::new("data/");
    let train_data: RepeatingNetworkData<BATCH_SIZE> =
        RepeatingNetworkData::new(mnist.train_labels, mnist.train_data);
    let test_data: RepeatingNetworkData<BATCH_SIZE> =
        RepeatingNetworkData::new(mnist.test_labels, mnist.test_data);

    // Build the network
    let network = build_network::<BATCH_SIZE>();

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
