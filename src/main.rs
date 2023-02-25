use rust_mnist::Mnist;

mod gradients;
mod network;
mod tensor_operations;
mod tensors;

use crate::network::{
    Network, NetworkMode, StandardClassificationNetworkHandler, StandardNetworkHandler,
};
use crate::tensors::{Tensor, Tensor1D};

const TRAINING_EPOCHS: usize = 200;
const EXAMPLES_PER_VALIDATION: usize = 10000;
const EXAMPLES_PER_EPOCH: usize = 20000;
const INPUTS: usize = 784;
const OUTPUTS: usize = 10;
const ADDITIONAL_STARTING_NODES: usize = 100;
const BATCH_SIZE: usize = 32;

// fn validate(network: &mut Network<BATCH_SIZE>, mnist: &Mnist) -> f64 {
//     network.set_mode(NetworkMode::Inference);
//     let mut correct: usize = 0;
//     for ii in 0..(EXAMPLES_PER_VALIDATION / BATCH_SIZE) {
//         // Prep the data
//         let start = ii * BATCH_SIZE;
//         let inputs: Vec<Tensor1D<BATCH_SIZE>> = (0..INPUTS)
//             .map(|i| {
//                 let data: [f64; BATCH_SIZE] = (start..start + BATCH_SIZE)
//                     .map(|ii| mnist.train_data[ii][i] as f64 / 255.)
//                     .collect::<Vec<f64>>()
//                     .try_into()
//                     .unwrap();
//                 Tensor1D::new_without_tape(data)
//             })
//             .collect();
//         let labels: Vec<usize> = (start..start + BATCH_SIZE)
//             .map(|i| mnist.train_labels[i] as usize)
//             .collect();
//         let outputs = network.forward_batch(inputs);
//         let guesses: Vec<usize> = (0..BATCH_SIZE)
//             .map(|i| {
//                 let mut max: (usize, f64) = (0, outputs[0].data[i]);
//                 for ii in 0..OUTPUTS {
//                     if outputs[ii].data[i] > max.1 {
//                         max = (ii, outputs[ii].data[i]);
//                     }
//                 }
//                 max.0
//             })
//             .collect();
//         correct += guesses
//             .iter()
//             .enumerate()
//             .filter(|(i, g)| **g == labels[*i])
//             .count();
//     }
//     (correct as f64) / (EXAMPLES_PER_VALIDATION - (EXAMPLES_PER_VALIDATION % BATCH_SIZE)) as f64
// }
//
// fn train_epoch_batch(network: &mut Network<BATCH_SIZE>, mnist: &Mnist) {
//     network.set_mode(NetworkMode::Training);
//     for ii in 0..(EXAMPLES_PER_EPOCH / BATCH_SIZE) {
//         // Prep the data
//         let start = ii * BATCH_SIZE;
//         let inputs: Vec<Tensor1D<BATCH_SIZE>> = (0..INPUTS)
//             .map(|i| {
//                 let data: [f64; BATCH_SIZE] = (start..start + BATCH_SIZE)
//                     .map(|ii| mnist.train_data[ii][i] as f64 / 255.)
//                     .collect::<Vec<f64>>()
//                     .try_into()
//                     .unwrap();
//                 Tensor1D::new_without_tape(data)
//             })
//             .collect();
//         let labels: Vec<usize> = (start..start + BATCH_SIZE)
//             .map(|i| mnist.train_labels[i] as usize)
//             .collect();
//
//         // Forward and bacward
//         let outputs = network.forward_batch(inputs);
//         let loss = &mut Tensor1D::nll(outputs, labels);
//         let grads = loss.backward();
//         network.apply_gradients(grads);
//     }
// }

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
    let mnist = Mnist::new("data/");
    let train_data: RepeatingNetworkData<BATCH_SIZE> =
        RepeatingNetworkData::new(mnist.train_labels, mnist.train_data);
    let test_data: RepeatingNetworkData<BATCH_SIZE> =
        RepeatingNetworkData::new(mnist.test_labels, mnist.test_data);

    let network = build_network::<BATCH_SIZE>();

    let mut network_handler = StandardClassificationNetworkHandler::new(
        network,
        1000000,
        2000,
        1000,
        Box::new(train_data),
        Box::new(test_data),
    );

    network_handler.train();

    // let mut network = build_network::<BATCH_SIZE>();
    // println!("{:?}", network);
    //
    // for i in 0..TRAINING_EPOCHS {
    //     train_epoch_batch(&mut network, &mnist);
    //     let percent_correct = validate(&mut network, &mnist);
    //
    //     // Print some nice things for us
    //     print!(
    //         "{}{}Epoch: {} val_percent: {}",
    //         termion::clear::CurrentLine,
    //         termion::cursor::Left(10000),
    //         i,
    //         percent_correct
    //     );
    //     println!();
    //     println!("{:?}", network);
    //     println!();
    // }
}
