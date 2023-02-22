use rust_mnist::Mnist;

mod gradients;
mod network;
mod tensor_operations;
mod tensors;

use crate::gradients::Gradients;
use crate::network::{Network, NetworkMode};
use crate::tensors::{Tensor, Tensor0D, Tensor1D};

const TRAINING_EPOCHS: usize = 200;
const EXAMPLES_PER_VALIDATION: usize = 1000;
const EXAMPLES_PER_EPOCH: usize = 2000;
const INPUTS: usize = 784;
const OUTPUTS: usize = 10;
const BATCH_SIZE: usize = 8;

fn validate(network: &mut Network, mnist: &Mnist) -> f64 {
    let mut correct = 0.;
    for i in 0..EXAMPLES_PER_VALIDATION {
        let input: Vec<Tensor0D> = mnist.test_data[i]
            .iter()
            .map(|x| Tensor0D::new_without_tape(*x as f64 / 255.))
            .collect();
        let label = mnist.test_labels[i] as usize;
        let output = network.forward(input);
        let guess = output
            .into_iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |acc, (i, t)| {
                if t.data > acc.1 {
                    (i, t.data)
                } else {
                    acc
                }
            })
            .0;
        if label == guess {
            correct += 1.;
        }
    }
    correct / EXAMPLES_PER_VALIDATION as f64
}

fn train_epoch(network: &mut Network, mnist: &Mnist) {
    network.set_mode(NetworkMode::Training);
    let mut merged_grads: Option<Gradients> = None;
    for ii in 0..EXAMPLES_PER_EPOCH {
        // Prep data
        let input: Vec<Tensor0D> = mnist.train_data[ii]
            .iter()
            .map(|x| Tensor0D::new_without_tape(*x as f64 / 255.))
            .collect();
        let label = mnist.train_labels[ii] as usize;

        // Forward and backward
        let output = network.forward(input);
        let loss = &mut Tensor0D::nll(output, label);
        let grads = loss.backward();
        match &mut merged_grads {
            Some(mm) => mm.merge_add(grads),
            None => merged_grads = Some(grads),
        }

        // Apply merged grads
        if ii % BATCH_SIZE == 0 {
            network.apply_gradients(merged_grads.unwrap(), 1.0);
            merged_grads = None;
        }
    }
}

fn train_epoch_batch(network: &mut Network, mnist: &Mnist) {
    network.set_mode(NetworkMode::Training);
    for ii in 0..(EXAMPLES_PER_EPOCH / BATCH_SIZE) {
        // Prep the data
        let start = ii * BATCH_SIZE;
        let inputs: Vec<Tensor1D<BATCH_SIZE>> = (0..INPUTS)
            .map(|i| {
                let data: [f64; BATCH_SIZE] = (start..start + BATCH_SIZE)
                    .map(|ii| mnist.train_data[ii][i] as f64 / 255.)
                    .collect::<Vec<f64>>()
                    .try_into()
                    .unwrap();
                Tensor1D::new_without_tape(data)
            })
            .collect();
        let labels: Vec<usize> = (start..start + BATCH_SIZE)
            .map(|i| mnist.train_labels[i] as usize)
            .collect();

        // Forward and bacward
        let outputs = network.forward_batch(inputs);
        let loss = &mut Tensor1D::nll(outputs, labels);
        let grads = loss.backward();
        network.apply_gradients(grads, 1.0);
    }
}

fn build_network() -> Network {
    let network = Network::new(INPUTS, 512, OUTPUTS);
    network
}

fn main() {
    let mnist = Mnist::new("data/");

    let mut network = build_network();
    println!("{:?}", network);

    for i in 0..TRAINING_EPOCHS {
        // train_epoch(&mut network, &mnist);
        train_epoch_batch(&mut network, &mnist);
        network.set_mode(NetworkMode::Inference);
        let percent_correct = validate(&mut network, &mnist);

        // Print some nice things for us
        print!(
            "{}{}Epoch: {} val_percent: {}",
            termion::clear::CurrentLine,
            termion::cursor::Left(10000),
            i,
            percent_correct
        );
        println!();
        println!("{:?}", network);
        println!();
    }
}
