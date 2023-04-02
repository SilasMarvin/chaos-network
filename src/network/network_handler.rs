use crate::build_order_network;
use crate::network::ChaosNetwork;
use crate::network::NodeKind;
use crate::tensors::Tensor1D;
use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;
use std::fs::DirBuilder;
use std::time::{SystemTime, UNIX_EPOCH};
use termion::input::TermRead;

use crate::network::HeadNetwork;
use crate::network::{OrderNetwork, OrderNetworkTrait};

const MORPHS_PER_ITERATION: usize = 1;

#[derive(Debug, PartialEq, Eq)]
enum Action {
    Stop,
    StopSave,
}

fn parse_input(inp: String) -> Result<(Action, f64), Box<dyn std::error::Error>> {
    let split: Vec<&str> = inp.split(" ").collect();
    if split.len() > 3 {
        return Err(Box::<dyn std::error::Error>::from(
            "Length cannot be greater than 2",
        ));
    }
    match split[0] {
        "s" => Ok((Action::Stop, 0.)),
        "ss" => Ok((Action::StopSave, 0.)),
        _ => return Err(Box::<dyn std::error::Error>::from("Action not found")),
    }
}

#[derive(Clone)]
pub struct RepeatingNetworkData<const I: usize, const N: usize> {
    labels: Vec<u8>,
    data: Vec<[u8; I]>,
}

impl<const I: usize, const N: usize> RepeatingNetworkData<I, N> {
    pub fn new(labels: Vec<u8>, data: Vec<[u8; I]>) -> Self {
        RepeatingNetworkData { labels, data }
    }

    fn next(&mut self) -> (Box<[usize; N]>, Box<[[f64; I]; N]>) {
        let mut rng = thread_rng();
        let distribution = Uniform::from(0..self.data.len());
        let indices: [usize; N] = (0..N)
            .map(|_i| distribution.sample(&mut rng))
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap();
        let inputs: [[f64; I]; N] =
            indices.map(|index| self.data[index].map(|d| (d as f64) / 255.));
        let labels: [usize; N] = indices.map(|index| self.labels[index] as usize);
        (Box::new(labels), Box::new(inputs))
    }
}

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

pub struct StandardClassificationNetworkHandler<
    const ON: usize,
    const OI: usize,
    const OO: usize,
    const CI: usize,
    const CO: usize,
    const O: usize,
    const N: usize,
> {
    max_training_steps: usize,
    steps_per_training_step: usize,
    train_data: RepeatingNetworkData<OI, N>,
    test_data: RepeatingNetworkData<OI, N>,
    validation_steps: usize,
    order_network: Option<Box<dyn OrderNetworkTrait<OI, OO, N>>>,
}

impl<
        const ON: usize,
        const OI: usize,
        const OO: usize,
        const CI: usize,
        const CO: usize,
        const O: usize,
        const N: usize,
    > StandardClassificationNetworkHandler<ON, OI, OO, CI, CO, O, N>
{
    pub fn new(
        max_training_steps: usize,
        steps_per_training_step: usize,
        train_data: RepeatingNetworkData<OI, N>,
        test_data: RepeatingNetworkData<OI, N>,
        validation_steps: usize,
        order_network: Option<Box<dyn OrderNetworkTrait<OI, OO, N>>>,
    ) -> Self {
        Self {
            max_training_steps,
            steps_per_training_step,
            train_data,
            test_data,
            validation_steps,
            order_network,
        }
    }
}

fn train_chaos_head_next_batch<
    const CI: usize,
    const CO: usize,
    const HO: usize,
    const N: usize,
>(
    chaos_network: &mut ChaosNetwork<CI, CO, N>,
    head_network: &mut HeadNetwork<CO, HO, N>,
    train_data: &(Box<[usize; N]>, Box<[Tensor1D<N>; CI]>),
) {
    let (labels, inputs) = train_data;
    let outputs = chaos_network.forward_batch(inputs);
    let output_ids: Vec<usize> = outputs.iter().map(|t| t.id).collect();
    let outputs = head_network.forward_batch_from_chaos(outputs);
    let (_losses, nll_grads) = HeadNetwork::<CO, HO, N>::nll(outputs, labels);
    let (head_network_grads, chaos_network_outputs_grads) = head_network.backwards(&nll_grads);
    head_network.apply_gradients(&head_network_grads);
    output_ids
        .into_iter()
        .zip(chaos_network_outputs_grads.into_iter())
        .for_each(|(id, grads)| {
            chaos_network.tape.add_operation((
                usize::MAX,
                Box::new(move |g| {
                    g.insert(id, Tensor1D::new(grads));
                }),
            ));
        });
    chaos_network.execute_and_apply_gradients();
}

// fn validate_chaos_head_next_batch<
//     const CI: usize,
//     const CO: usize,
//     const HO: usize,
//     const N: usize,
// >(
//     chaos_network: &mut ChaosNetwork<CI, CO, N>,
//     head_network: &mut HeadNetwork<CO, HO, N>,
//     test_data: &(Box<[usize; N]>, Box<[Tensor1D<N>; CI]>),
// ) -> f64 {
//     let (labels, inputs) = test_data;
//     let outputs = chaos_network.forward_batch_no_grad(inputs);
//     let outputs = head_network.forward_batch_no_grad_from_chaos(outputs);
//     let guesses: Vec<usize> = (0..N)
//         .map(|i| {
//             let mut max: (usize, f64) = (0, outputs[0][i]);
//             for ii in 0..HO {
//                 if outputs[ii][i] > max.1 {
//                     max = (ii, outputs[ii][i]);
//                 }
//             }
//             max.0
//         })
//         .collect();
//     let correct = guesses
//         .iter()
//         .enumerate()
//         .filter(|(i, g)| **g == labels[*i])
//         .count();
//     (correct as f64) / N as f64
// }

fn validate_chaos_head_next_batch<
    const CI: usize,
    const CO: usize,
    const HO: usize,
    const N: usize,
>(
    chaos_network: &mut ChaosNetwork<CI, CO, N>,
    head_network: &mut HeadNetwork<CO, HO, N>,
    test_data: &(Box<[usize; N]>, Box<[Tensor1D<N>; CI]>),
) -> f64 {
    let (labels, inputs) = test_data;
    let outputs = chaos_network.forward_batch_no_grad(inputs);
    let outputs = head_network.forward_batch_no_grad(outputs);
    let correct = labels
        .into_iter()
        .enumerate()
        .map(|(i, correct_label_index)| {
            let guess = outputs[i]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            guess.0 == correct_label_index
        })
        .filter(|v| *v)
        .count();
    (correct as f64) / N as f64
}

// fn train_next_batch<const CI: usize, const CO: usize, const HO: usize, const N: usize>(
//     order_network: &mut Option<Box<dyn OrderNetworkTrait<OI, OO, N>>>,
//     chaos_network: &mut ChaosNetwork<CI, CO, N>,
//     head_network: &mut HeadNetwork<CO, HO, N>,
//     train_data: &(Box<[usize; N]>, Box<[Tensor1D<N>; CI]>),
// ) {
//     let (labels, inputs) = train_data;
//     let outputs = chaos_network.forward_batch(inputs);
//     let output_ids: Vec<usize> = outputs.iter().map(|t| t.id).collect();
//     let outputs = head_network.forward_batch(outputs);
//     let (_losses, nll_grads) = HeadNetwork::<CO, HO, N>::nll(outputs, labels);
//     let (head_network_grads, chaos_network_outputs_grads) = head_network.backwards(&nll_grads);
//     head_network.apply_gradients(&head_network_grads);
//     output_ids
//         .into_iter()
//         .zip(chaos_network_outputs_grads.into_iter())
//         .for_each(|(id, grads)| {
//             chaos_network.tape.add_operation((
//                 usize::MAX,
//                 Box::new(move |g| {
//                     g.insert(id, Tensor1D::new(grads));
//                 }),
//             ));
//         });
//     chaos_network.execute_and_apply_gradients();
// }
//
// fn validate_next_batch<const CI: usize, const CO: usize, const HO: usize, const N: usize>(
//     order_network: &mut Option<Box<dyn OrderNetworkTrait<OI, OO, N>>>,
//     chaos_network: &mut ChaosNetwork<CI, CO, N>,
//     head_network: &mut HeadNetwork<CO, HO, N>,
//     test_data: &(Box<[usize; N]>, Box<[Tensor1D<N>; CI]>),
// ) -> f64 {
//     let (labels, inputs) = test_data;
//     let outputs = chaos_network.forward_batch_no_grad(inputs);
//     let outputs = head_network.forward_batch_no_grad(outputs);
//     let guesses: Vec<usize> = (0..N)
//         .map(|i| {
//             let mut max: (usize, f64) = (0, outputs[0][i]);
//             for ii in 0..HO {
//                 if outputs[ii][i] > max.1 {
//                     max = (ii, outputs[ii][i]);
//                 }
//             }
//             max.0
//         })
//         .collect();
//     let correct = guesses
//         .iter()
//         .enumerate()
//         .filter(|(i, g)| **g == labels[*i])
//         .count();
//     (correct as f64) / N as f64
// }

impl<
        const ON: usize,
        const OI: usize,
        const OO: usize,
        const CI: usize,
        const CO: usize,
        const O: usize,
        const N: usize,
    > StandardClassificationNetworkHandler<ON, OI, OO, CI, CO, O, N>
{
    pub fn train_chaos_head(&mut self) {
        let mut stdin = termion::async_stdin();
        let mut current_chaos_network: ChaosNetwork<OI, CO, N> = ChaosNetwork::new(CI, CO);
        let mut current_head_network: HeadNetwork<CO, O, N> = HeadNetwork::default();

        // Do the actual training
        for training_step in 0..self.max_training_steps {
            // Prep the data
            let batch_train_data: Vec<(Box<[usize; N]>, Box<[Tensor1D<N>; OI]>)> = (0..self
                .steps_per_training_step)
                .map(|_i| {
                    let (train_labels, train_examples) = self.train_data.next();
                    (
                        train_labels,
                        transform_train_data_for_chaos_network(train_examples),
                    )
                })
                .collect();
            let batch_test_data: Vec<(Box<[usize; N]>, Box<[Tensor1D<N>; OI]>)> = (0..self
                .validation_steps)
                .map(|_i| {
                    let (train_labels, train_examples) = self.test_data.next();
                    (
                        train_labels,
                        transform_train_data_for_chaos_network(train_examples),
                    )
                })
                .collect();

            // Prep the nodes and edges to add
            let nodes_to_add =
                ((current_chaos_network.get_normal_node_count() as f64 * 0.05) as usize).max(5);
            let edges_to_add =
                ((current_chaos_network.get_edge_count() as f64 * 0.05) as usize).max(25);

            // Do it!
            (current_chaos_network, current_head_network) = if training_step % 10 == 0 {
                let new_networks: Vec<(ChaosNetwork<OI, CO, N>, HeadNetwork<CO, O, N>, f64)> = (0
                    ..MORPHS_PER_ITERATION)
                    .into_par_iter()
                    .map(|_i| {
                        let mut chaos_network = if training_step != 0 {
                            let mut network = current_chaos_network.clone();
                            network.add_nodes(NodeKind::Normal, nodes_to_add);
                            network.add_random_edges_for_random_nodes(edges_to_add);
                            network
                        } else {
                            // ChaosNetwork::new(OI, CO)
                            current_chaos_network.clone()
                        };
                        let mut head_network = current_head_network.clone();
                        println!("WE ARE HERE");
                        batch_train_data.iter().for_each(|batch| {
                            train_chaos_head_next_batch(
                                &mut chaos_network,
                                &mut head_network,
                                batch,
                            )
                        });
                        println!("No way we got here");
                        let average_validation_accuracy: f64 = batch_test_data
                            .iter()
                            .map(|step| {
                                validate_chaos_head_next_batch(
                                    &mut chaos_network,
                                    &mut head_network,
                                    step,
                                )
                            })
                            .sum::<f64>()
                            / (self.validation_steps as f64);
                        (chaos_network, head_network, average_validation_accuracy)
                    })
                    .collect();
                let (new_chaos_network, new_head_network, average_validation_accuracy): (
                    ChaosNetwork<OI, CO, N>,
                    HeadNetwork<CO, O, N>,
                    f64,
                ) = new_networks
                    .into_iter()
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                    .unwrap();
                println!(
                    "AVA: {}{:?}",
                    average_validation_accuracy, new_chaos_network
                );
                (new_chaos_network, new_head_network)
            } else {
                batch_train_data.iter().for_each(|batch| {
                    train_chaos_head_next_batch(
                        &mut current_chaos_network,
                        &mut current_head_network,
                        batch,
                    )
                });
                (current_chaos_network, current_head_network)
            };

            // Match user input
            if let Some(line) = stdin.read_line().unwrap() {
                if line.len() > 0 {
                    let x = parse_input(line);
                    match x {
                        Ok((action, num1)) => match action {
                            Action::Stop => return,
                            Action::StopSave => {
                                let path: String = format!(
                                    "networks/{}",
                                    SystemTime::now()
                                        .duration_since(UNIX_EPOCH)
                                        .unwrap()
                                        .as_secs()
                                );
                                DirBuilder::new().recursive(true).create(&path).unwrap();
                                current_chaos_network
                                    .write_to_dir(&path)
                                    .unwrap_or_else(|_err| panic!("Error saving chaos network"));
                                current_head_network
                                    .write_to_dir(&path)
                                    .unwrap_or_else(|_err| panic!("Error saving chaos network"));
                                println!("Chaos and Head Network saved");
                                return;
                            }
                        },
                        Err(error) => println!("ERROR: {:?}", error),
                    }
                }
            }
        }
    }

    // pub fn fine_tune(&mut self) {
    //     let mut stdin = termion::async_stdin();
    //     let mut current_order_network: OrderNetworkTrait<OI, OO, N> = self.order_network.unwrap();
    //     let mut current_head_network: HeadNetwork<OO, O, N> = HeadNetwork::default();
    //
    //     // Do the actual training
    //     for training_step in 0..self.max_training_steps {
    //         // Prep the data
    //         let batch_train_data: Vec<(Box<[usize; N]>, Box<[Tensor1D<N>; OI]>)> = (0..self
    //             .steps_per_training_step)
    //             .map(|_i| {
    //                 let (train_labels, train_examples) = self.train_data.next();
    //                 (train_labels, train_examples)
    //             })
    //             .collect();
    //         let batch_test_data: Vec<(Box<[usize; N]>, Box<[Tensor1D<N>; OI]>)> = (0..self
    //             .validation_steps)
    //             .map(|_i| {
    //                 let (train_labels, train_examples) = self.test_data.next();
    //                 (train_labels, train_examples)
    //             })
    //             .collect();
    //
    //         batch_train_data.iter().for_each(|batch| {
    //             train_next_batch(
    //                 &mut current_order_network,
    //                 &mut current_chaos_network,
    //                 &mut current_head_network,
    //                 batch,
    //             )
    //         });
    //     }
    // }
}
