use rust_mnist::Mnist;
use std::io::{stdout, Write};

mod gradients;
mod network;
mod tensor_operations;
mod tensors;

use crate::network::{Network, NetworkMode, NodeKind};
use crate::tensors::Tensor0D;

const TRAINING_EPOCHS: usize = 50;
const EXAMPLES_PER_EPOCH: usize = 10000;
const INPUTS: usize = 784;
const OUTPUTS: usize = 10;
const STARTING_NODES: usize = 5000;

fn validate(network: &mut Network, mnist: &Mnist) -> f32 {
    let mut correct = 0;
    for i in 0..10000 {
        let input: Vec<Tensor0D> = mnist.test_data[i]
            .iter()
            .map(|x| Tensor0D::new_without_tape(*x as f32 / 255.))
            .collect();
        let label = mnist.test_labels[i] as usize;
        let output = network.forward(input);
        let guess = output
            .into_iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |acc, (i, t)| {
                if t.data > acc.1 {
                    (i, t.data)
                } else {
                    acc
                }
            })
            .0;
        if label == guess {
            correct += 1;
        }
    }
    correct as f32 / 10000.
}

fn build_network() -> Network {
    let mut network = Network::new();
    for _i in 0..INPUTS {
        network.add_node(NodeKind::Input);
    }
    for _i in 0..OUTPUTS {
        network.add_node(NodeKind::Leaf);
    }
    for _i in 0..STARTING_NODES {
        network.add_node(NodeKind::Normal);
    }
    network
}

fn main() {
    let mnist = Mnist::new("data/");

    let mut network = build_network();
    println!("{:?}", network);

    // Do initial validation
    network.set_mode(NetworkMode::Inference);
    validate(&mut network, &mnist);
    network.set_mode(NetworkMode::Training);

    for i in 0..TRAINING_EPOCHS {
        for ii in 0..EXAMPLES_PER_EPOCH {
            // Prep data
            let input: Vec<Tensor0D> = mnist.train_data[ii]
                .iter()
                .map(|x| Tensor0D::new_without_tape(*x as f32 / 255.))
                .collect();
            let label = mnist.train_labels[ii] as usize;

            // Forward pass
            let output = network.forward(input);
            let loss = Tensor0D::nll(output, label);
            network.backward(loss);

            // Print some nice things for us
            if ii % 100 == 0 {
                let percent_done = ii as f32 / EXAMPLES_PER_EPOCH as f32;
                let mut progress = "#".repeat((percent_done * 100.) as usize);
                progress += &" ".repeat(((1. - percent_done) * 100.) as usize);
                print!(
                    "{}{}Epoch: {} [{}] {}/{}",
                    termion::clear::CurrentLine,
                    termion::cursor::Left(10000),
                    i,
                    progress,
                    ii,
                    EXAMPLES_PER_EPOCH
                );
                stdout().flush().unwrap();
            }
        }
        // Do end of epoch validation
        network.set_mode(NetworkMode::Inference);
        let percent_correct = validate(&mut network, &mnist);
        network.set_mode(NetworkMode::Training);
        print!(
            "{}{}Epoch: {} val_percent: {}",
            termion::clear::CurrentLine,
            termion::cursor::Left(10000),
            i,
            percent_correct
        );
        println!();
    }
}
