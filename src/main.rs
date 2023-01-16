use device_query::{DeviceQuery, DeviceState, Keycode};
use rust_mnist::Mnist;

use std::io::Write;

mod gradients;
mod network;
mod tensor_operations;
mod tensors;

use crate::network::{Network, NetworkMode, NodeKind};
use crate::tensors::Tensor0D;

const BATCH_SIZE: usize = 8;

const INPUTS: usize = 784;
const OUTPUTS: usize = 10;
const STARTING_NODES: usize = 5000;

fn validate(network: &mut Network, mnist: &Mnist) {
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
    println!("VAL PERCENT: {:?}", correct as f32 / 10000 as f32);
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
    network.add_random_connections(10000);

    println!("{:?}", network);

    let device_state = DeviceState::new();
    let mut running_loss = Tensor0D::new_without_tape(0.);
    network.set_mode(NetworkMode::Training);
    for i in 0..1000 {
        for ii in 0..60000 {
            // Prep data
            let input: Vec<Tensor0D> = mnist.train_data[ii]
                .iter()
                .map(|x| Tensor0D::new_without_tape(*x as f32 / 255.))
                .collect();
            let label = mnist.train_labels[ii] as usize;

            // Forward pass
            let output = network.forward(input);
            let mut loss = Tensor0D::nll(output, label);

            // Update
            running_loss = &mut running_loss + &mut loss;

            if ii % BATCH_SIZE == 0 {
                running_loss =
                    &mut running_loss * &mut Tensor0D::new_without_tape(1. / BATCH_SIZE as f32);
                if ii % (BATCH_SIZE * 1000) == 0 {
                    println!("Iteration: {} - loss: {:?}", i, running_loss);
                }
                network.backward(running_loss);
                running_loss = Tensor0D::new_without_tape(0.);
            }

            // Check to update the model
            let keys: Vec<Keycode> = device_state.get_keys();
            if keys.contains(&Keycode::Space) {
                println!("What do you want to do?");
                println!("\tPrune: p");
                println!("\tAdd Nodes: n");
                println!("\tAdd Connections: c");
                print!("Enter a choice: ");
                std::io::stdout().flush().unwrap();
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                match line.as_str().trim() {
                    "p" => {
                        network.prune_weights_below(0.00001);
                        println!("Pruned");
                        println!("{:?}", network);
                    }
                    "n" => {
                        for _i in 0..1000 {
                            network.add_node(NodeKind::Normal);
                        }
                        println!("Added Nodes");
                        println!("{:?}", network);
                    }
                    "c" => {
                        network.add_random_connections(1000);
                        println!("Added Connections");
                        println!("{:?}", network);
                    }
                    _ => continue,
                }
            }
        }
        network.set_mode(NetworkMode::Inference);
        validate(&mut network, &mnist);
        network.set_mode(NetworkMode::Training);
    }
}
