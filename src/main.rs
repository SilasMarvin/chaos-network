use crate::network::ChaosNetwork;
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
const STEPS_PER_TRAINING_STEPS: usize = 400;
const VALIDATION_STEPS: usize = 150;

// Order Network build constants
const ON: usize = 1680470380;
const OI: usize = INPUTS;
const OO: usize = 500;

// Chaos network build constants
const CI: usize = INPUTS;
const CO: usize = 500;

use crate::tensors::Tensor1D;
fn transform_train_data_for_chaos_network<const I: usize, const N: usize>(
    data: Box<[[f64; I]; N]>,
) -> Box<[Tensor1D<N>; I]> {
    let ret: [Tensor1D<N>; I] = (0..I)
        .map(|i| {
            let t_data: [f64; N] = (0..N)
                .map(|ii| data[ii][i])
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();
            Tensor1D::new(t_data)
        })
        .collect::<Vec<Tensor1D<N>>>()
        .try_into()
        .unwrap();
    Box::new(ret)
}

fn main() {
    // let inputs = Box::new([[0.1; 10]; 1]);

    // let inputs = transform_train_data_for_chaos_network(inputs);
    // let mut chaos_network: ChaosNetwork<10, 10, 1> = ChaosNetwork::new(10, 10);
    // chaos_network.write_to_dir("networks/10").unwrap();
    // let outputs = chaos_network.forward_batch(&inputs);

    // build_order_network!(10, 10, 10, 1);
    // let mut order_network: OrderNetwork<10, 10, 10, 1> = OrderNetwork::default();
    // let outputs = order_network.forward_batch(inputs);
    // println!("Weights: {:?}", order_network.weights);
    //
    // println!("Outputs: {:?}", outputs);

    build_order_network!(1680549994, 3072, 500, 32);
    let order_network: OrderNetwork<1680549994, 3072, 500, BATCH_SIZE> = OrderNetwork::default();

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
        Some(Box::new(order_network)),
    );

    // Train
    // network_handler.train_chaos_head();
    network_handler.fine_tune();

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
