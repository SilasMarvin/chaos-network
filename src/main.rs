use cifar_ten::*;

mod gradients;
mod network;
mod tensor_operations;
mod tensors;

use crate::network::{RepeatingNetworkData, StandardClassificationNetworkHandler};

// General constants
const INPUTS: usize = 3072;
const OUTPUTS: usize = 10;
const BATCH_SIZE: usize = 32;
const MAX_TRAINING_STEPS: usize = 1000000;
const STEPS_PER_TRAINING_STEPS: usize = 150;
const VALIDATION_STEPS: usize = 75;

// Order Network build constants
const ON: usize = 1680044561;
const OI: usize = INPUTS;
const OO: usize = 500;

// Chaos network build constants
const CI: usize = OI;
const CO: usize = 500;

fn main() {
    // build_order_network!(0, 3072, 500, 32);
    // let mut current_order_network: OrderNetwork<0, 3072, 500, BATCH_SIZE> = OrderNetwork::default();

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
    let mut network_handler: StandardClassificationNetworkHandler<
        ON,
        OI,
        OO,
        CI,
        CO,
        OUTPUTS,
        BATCH_SIZE,
    > = StandardClassificationNetworkHandler::new(
        MAX_TRAINING_STEPS,
        STEPS_PER_TRAINING_STEPS,
        train_data,
        test_data,
        VALIDATION_STEPS,
    );

    // Train
    network_handler.train();

    //
    //
    //
    //
    // // Load data
    // let CifarResult(train_data, train_labels, test_data, test_labels) = Cifar10::default()
    //     .download_and_extract(true)
    //     .download_url("https://cmoran.xyz/data/cifar/cifar-10-binary.tar.gz")
    //     .encode_one_hot(false)
    //     .build()
    //     .unwrap();
    //
    // let train_data = train_data
    //     .chunks(INPUTS)
    //     .map(|v| {
    //         let x: [u8; INPUTS] = v
    //             .to_owned()
    //             .try_into()
    //             .expect("Error converting dataset u8 to f64");
    //         x
    //     })
    //     .collect::<Vec<[u8; INPUTS]>>();
    //
    // let test_data = test_data
    //     .chunks(INPUTS)
    //     .map(|v| {
    //         let x: [u8; INPUTS] = v
    //             .to_owned()
    //             .try_into()
    //             .expect("Error converting dataset u8 to f64");
    //         x
    //     })
    //     .collect::<Vec<[u8; INPUTS]>>();
    //
    // let train_data: RepeatingNetworkData<INPUTS, BATCH_SIZE> =
    //     RepeatingNetworkData::new(train_labels, train_data);
    // let test_data: RepeatingNetworkData<INPUTS, BATCH_SIZE> =
    //     RepeatingNetworkData::new(test_labels, test_data);
    //
    // // Build the network handler
    // let mut network_handler: StandardClassificationNetworkHandler<INPUTS, OUTPUTS, BATCH_SIZE> =
    //     StandardClassificationNetworkHandler::new(
    //         MAX_TRAINING_STEPS,
    //         STEPS_PER_TRAINING_STEPS,
    //         train_data,
    //         test_data,
    //         VALIDATION_STEPS,
    //     );
    //
    // // Train
    // network_handler.train();
}
