use cifar_ten::*;

mod gradients;
mod network;
mod tensor_operations;
mod tensors;

use crate::network::{RepeatingNetworkData, StandardClassificationNetworkHandler};

const INPUTS: usize = 3072;
const OUTPUTS: usize = 10;
const BATCH_SIZE: usize = 32;
const MAX_TRAINING_STEPS: usize = 50;
const STEPS_PER_TRAINING_STEPS: usize = 500;
const VALIDATION_STEPS: usize = 100;

fn main() {
    // Load data
    let CifarResult(train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .download_and_extract(true)
        .download_url("https://cmoran.xyz/data/cifar/cifar-10-binary.tar.gz")
        .encode_one_hot(false)
        .build()
        .unwrap();

    let train_data = train_data
        .chunks(INPUTS)
        .map(|v| {
            let x: [u8; INPUTS] = v
                .to_owned()
                .try_into()
                .expect("Error converting dataset u8 to f64");
            x
        })
        .collect::<Vec<[u8; INPUTS]>>();

    let test_data = test_data
        .chunks(INPUTS)
        .map(|v| {
            let x: [u8; INPUTS] = v
                .to_owned()
                .try_into()
                .expect("Error converting dataset u8 to f64");
            x
        })
        .collect::<Vec<[u8; INPUTS]>>();

    let train_data: RepeatingNetworkData<INPUTS, BATCH_SIZE> =
        RepeatingNetworkData::new(train_labels, train_data);
    let test_data: RepeatingNetworkData<INPUTS, BATCH_SIZE> =
        RepeatingNetworkData::new(test_labels, test_data);

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
    let now = std::time::Instant::now();
    network_handler.train();
    let elapsed_time = now.elapsed();
    println!("Elapsed: {:?} seconds", elapsed_time.as_secs());
}
