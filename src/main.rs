use rand::prelude::*;
use rand::seq::SliceRandom;
use rust_mnist::Mnist;
use std::io::{stdout, Write};
use std::sync::Arc;
use std::thread;

mod gradients;
mod network;
mod tensor_operations;
mod tensors;

use crate::gradients::Gradients;
use crate::network::{Network, NetworkMode, NodeKind};
use crate::tensors::{Tensor, Tensor0D};

const WORKERS_COUNT: usize = 1;

const TRAINING_EPOCHS: usize = 5000;
const EXAMPLES_PER_VALIDATION: usize = 1000;
const EXAMPLES_PER_EPOCH: usize = 20000;
const INPUTS: usize = 784;
const OUTPUTS: usize = 10;
const ADDITIONAL_STARTING_NODES: i32 = 500;

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
    let mut merged_grads: Option<Gradients> = None;
    for ii in 0..EXAMPLES_PER_EPOCH {
        // Reset tapes
        network.set_mode(NetworkMode::Training);

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
        if ii % 32 == 0 {
            network.apply_gradients(merged_grads.unwrap(), 1.0);
            merged_grads = None;
        }
    }
}

fn build_network() -> Network {
    let network = Network::new(INPUTS, 512, OUTPUTS);
    network
}

fn main() {
    let mut mnist = Mnist::new("data/");
    let mut rng = rand::thread_rng();

    let mut network = build_network();
    println!("{:?}", network);

    for i in 0..TRAINING_EPOCHS {
        // mnist.train_data.shuffle(&mut rng);
        // let mut handles = Vec::new();
        // for _i in 0..WORKERS_COUNT {
        //     network.set_mode(NetworkMode::Training);
        //     let mut new_network = network.clone();
        //     let local_mnist = mnist.clone();
        //     let handle = thread::spawn(move || {
        //         // new_network.morph();
        //         println!("Training");
        //         train_epoch(&mut new_network, &local_mnist);
        //         new_network.set_mode(NetworkMode::Inference);
        //         let percent_correct = validate(&mut new_network, &local_mnist);
        //         (percent_correct, new_network)
        //     });
        //     handles.push(handle);
        // }
        //
        // let (percent_correct, new_network) = handles
        //     .into_iter()
        //     .map(|h| h.join().unwrap())
        //     .inspect(|h| println!("{:?}", h.0))
        //     .max_by(|x, y| x.0.total_cmp(&y.0))
        //     .unwrap();
        // network = new_network;

        network.set_mode(NetworkMode::Training);
        train_epoch(&mut network, &mnist);
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

    // for i in 0..TRAINING_EPOCHS {
    //     for ii in 0..EXAMPLES_PER_EPOCH {
    //         // Prep data
    //         let input: Vec<Tensor0D> = mnist.train_data[ii]
    //             .iter()
    //             .map(|x| Tensor0D::new_without_tape(*x as f64 / 255.))
    //             .collect();
    //         let label = mnist.train_labels[ii] as usize;
    //
    //         // Forward and backward
    //         let output = network.forward(input);
    //         let loss = Tensor0D::nll(output, label);
    //         network.backward(loss);
    //
    //         // Print some nice things for us
    //         if ii % 100 == 0 {
    //             let percent_done = ii as f64 / EXAMPLES_PER_EPOCH as f64;
    //             let mut progress = "#".repeat((percent_done * 100.) as usize);
    //             progress += &" ".repeat(((1. - percent_done) * 100.) as usize);
    //             print!(
    //                 "{}{}Epoch: {} [{}] {}/{}",
    //                 termion::clear::CurrentLine,
    //                 termion::cursor::Left(10000),
    //                 i,
    //                 progress,
    //                 ii,
    //                 EXAMPLES_PER_EPOCH
    //             );
    //             stdout().flush().unwrap();
    //         };
    //     }
    //     // Do end of epoch validation
    //     network.set_mode(NetworkMode::Inference);
    //     let percent_correct = validate(&mut network, &mnist);
    //     network.morph();
    //     network.set_mode(NetworkMode::Training);
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
