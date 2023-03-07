use rust_mnist::Mnist;
use std::sync::Arc;

mod gradients;
mod network;
mod tensor_operations;
mod tensors;

use crate::network::{RepeatingNetworkData, StandardClassificationNetworkHandler};

const INPUTS: usize = 784;
const OUTPUTS: usize = 10;
const BATCH_SIZE: usize = 32;
const MAX_TRAINING_STEPS: usize = 1000000;
const STEPS_PER_TRAINING_STEPS: usize = 500;
const VALIDATION_STEPS: usize = 100;

fn main() {
    // Load data
    let mnist = Mnist::new("data/");
    let train_data: RepeatingNetworkData<INPUTS, BATCH_SIZE> =
        RepeatingNetworkData::new(mnist.train_labels, mnist.train_data);
    let test_data: RepeatingNetworkData<INPUTS, BATCH_SIZE> =
        RepeatingNetworkData::new(mnist.test_labels, mnist.test_data);

    // Build the network handler
    let mut network_handler: StandardClassificationNetworkHandler<INPUTS, OUTPUTS, BATCH_SIZE> =
        StandardClassificationNetworkHandler::new(
            MAX_TRAINING_STEPS,
            STEPS_PER_TRAINING_STEPS,
            train_data,
            test_data,
            VALIDATION_STEPS,
        );

    // Train
    network_handler.train();
}
