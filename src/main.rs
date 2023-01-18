use rust_mnist::Mnist;

mod gradients;
mod network;
mod tensor_operations;
mod tensors;

use crate::network::{Network, NetworkMode, NodeKind};
use crate::tensors::Tensor0D;

const BATCH_SIZE: usize = 32;

const INPUTS: usize = 784;
const OUTPUTS: usize = 10;
const STARTING_NODES: usize = 5000;

fn validate(network: &mut Network, mnist: &Mnist) {
    let mut correct = 0;
    for i in 0..1000 {
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
    println!("VAL PERCENT: {:?}", correct as f32 / 1000 as f32);
}

fn main() {
    let mnist = Mnist::new("data/");

    let mut network = Network::new();
    network.add_node(NodeKind::Input);
    for _i in 0..OUTPUTS {
        network.add_node(NodeKind::Leaf);
    }
    for _i in 0..INPUTS - 1 {
        network.add_node(NodeKind::Input);
    }
    for _i in 0..STARTING_NODES {
        network.add_node(NodeKind::Normal);
    }

    println!("{:?}", network);

    println!("Initial test");
    network.set_mode(NetworkMode::Inference);
    validate(&mut network, &mnist);
    network.set_mode(NetworkMode::Training);

    let mut running_loss = Tensor0D::new_without_tape(0.);
    for i in 0..1000 {
        for ii in 0..60000 {
            network.set_mode(NetworkMode::Training);
            // Prep data
            let input: Vec<Tensor0D> = mnist.train_data[ii]
                .iter()
                .map(|x| Tensor0D::new_without_tape(*x as f32 / 255.))
                .collect();
            let label = mnist.train_labels[ii] as usize;

            // Forward pass
            let output = network.forward(input);
            let mut loss = Tensor0D::nll(output, label);

            // Update running loss
            running_loss = &mut running_loss + &mut loss;

            if ii % BATCH_SIZE == 0 {
                running_loss =
                    &mut running_loss * &mut Tensor0D::new_without_tape(1. / BATCH_SIZE as f32);
                if ii % (BATCH_SIZE * 50) == 0 {
                    println!("Iteration: {} - loss: {:?}", i, running_loss);
                    network.morph(1.);
                    network.grow(1.);
                }
                network.backward(running_loss);
                running_loss = Tensor0D::new_without_tape(0.);
            }
        }
        network.set_mode(NetworkMode::Inference);
        validate(&mut network, &mnist);
        network.set_mode(NetworkMode::Training);
    }
}
