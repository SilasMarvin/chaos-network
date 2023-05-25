use mnist::*;

mod network;
use crate::network::network_handler::{RepeatingNetworkData, StandardClassificationNetworkHandler};
use chaos_network_derive::get_weights_count;

// General constants
const INPUTS: usize = 784;
const OUTPUTS: usize = 10;
const BATCH_SIZE: usize = 64;
const MAX_TRAINING_STEPS: usize = 1000000;
const STEPS_PER_TRAINING_STEPS: usize = 400;
const VALIDATION_STEPS: usize = 150;

// Order Network build constants
const ON: usize = 0; // This does nothing right now
const OI: usize = INPUTS;
const OO: usize = 500;

// Chaos network build constants
const CI: usize = 500; // Only used when training the order and chaos and head
const CO: usize = 500;

fn main() {
    const RWEIGHTS_COUNT: usize = get_weights_count!(1681273119);
    build_order_network!(1681273119, 784, 500, 64);
    let order_network: OrderNetwork<1681273119, RWEIGHTS_COUNT, 784, 500, BATCH_SIZE> =
        OrderNetwork::default();
    let order_network: Option<Box<dyn OrderNetworkTrait<OI, OO, BATCH_SIZE>>> =
        Some(Box::new(order_network));
    // let order_network = None;

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_data = trn_img
        .chunks(INPUTS)
        .map(|v| {
            let x: [u8; INPUTS] = v
                .to_owned()
                .try_into()
                .expect("Error converting dataset u8 to f64");
            x
        })
        .collect::<Vec<[u8; INPUTS]>>();
    let train_labels = trn_lbl;

    let test_data = tst_img
        .chunks(INPUTS)
        .map(|v| {
            let x: [u8; INPUTS] = v
                .to_owned()
                .try_into()
                .expect("Error converting dataset u8 to f64");
            x
        })
        .collect::<Vec<[u8; INPUTS]>>();
    let test_labels = tst_lbl;

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
        order_network,
    );

    // Train
    // network_handler.train_chaos_head();
    // network_handler.train_order_chaos_head();
    network_handler.fine_tune();
}
