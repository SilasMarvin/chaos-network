use crate::network::ChaosNetwork;
use crate::tensor_operations::Tensor1DNll;
use crate::tensors::Tensor1D;
use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;
use termion::input::TermRead;

use crate::network::HeadNetwork;

const MORPHS_PER_ITERATION: usize = 1;
const CHAOS_NETWORK_OUTPUTS: usize = 100;

#[derive(Debug, PartialEq, Eq)]
enum Action {
    ChangeSizeWeight,
}

fn parse_input(inp: String) -> Result<(Action, f64), Box<dyn std::error::Error>> {
    let split: Vec<&str> = inp.split(" ").collect();
    if split.len() > 3 {
        return Err(Box::<dyn std::error::Error>::from(
            "Length cannot be greater than 2",
        ));
    }
    match split[0] {
        "w" => {
            if split.len() < 2 {
                return Err(Box::<dyn std::error::Error>::from(
                    "The ChangeSizeWeight action requires 2 inputs",
                ));
            }
            let num1 = split[1].parse::<f64>()?;
            Ok((Action::ChangeSizeWeight, num1))
        }
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

pub struct StandardClassificationNetworkHandler<const I: usize, const O: usize, const N: usize> {
    max_training_steps: usize,
    steps_per_training_step: usize,
    train_data: RepeatingNetworkData<I, N>,
    test_data: RepeatingNetworkData<I, N>,
    validation_steps: usize,
}

impl<const I: usize, const O: usize, const N: usize> StandardClassificationNetworkHandler<I, O, N> {
    pub fn new(
        max_training_steps: usize,
        steps_per_training_step: usize,
        train_data: RepeatingNetworkData<I, N>,
        test_data: RepeatingNetworkData<I, N>,
        validation_steps: usize,
    ) -> Self {
        Self {
            max_training_steps,
            steps_per_training_step,
            train_data,
            test_data,
            validation_steps,
        }
    }
}

fn grow<const I: usize, const O: usize, const N: usize>(
    network: &mut ChaosNetwork<I, O, N>,
    percent_nodes_to_add: f64,
    percent_edges_to_add: f64,
) {
    let nodes_to_add = (network.nodes.len() as f64 * percent_nodes_to_add) as usize;
    let edges_to_add = (network.get_edge_count() as f64 * percent_edges_to_add) as usize;
    network.add_nodes(super::NodeKind::Normal, nodes_to_add);
    network.add_random_edges_for_random_nodes(edges_to_add);
}

fn prune<const I: usize, const O: usize, const N: usize>(
    network: &mut ChaosNetwork<I, O, N>,
    percent_edges_to_remove: f64,
) {
    let edges_to_remove = (network.get_edge_count() as f64 * percent_edges_to_remove) as usize;
    network.remove_weighted_edges(edges_to_remove);
}

fn train_next_batch<const CI: usize, const CO: usize, const HO: usize, const N: usize>(
    chaos_network: &mut ChaosNetwork<CI, CO, N>,
    head_network: &mut HeadNetwork<CO, HO, N>,
    train_data: &(Box<[usize; N]>, Box<[Tensor1D<N>; CI]>),
) {
    let (labels, inputs) = train_data;
    let outputs = chaos_network.forward_batch(inputs);
    let output_ids: Vec<usize> = outputs.iter().map(|t| t.id).collect();
    let outputs = head_network.forward_batch(outputs);
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

    // let outputs = chaos_network.forward_batch(inputs);
    // let _loss = &mut Tensor1D::nll(outputs, labels, &mut chaos_network.tape);
    // chaos_network.execute_and_apply_gradients();
}

fn validate_next_batch<const CI: usize, const CO: usize, const HO: usize, const N: usize>(
    chaos_network: &mut ChaosNetwork<CI, CO, N>,
    head_network: &mut HeadNetwork<CO, HO, N>,
    test_data: &(Box<[usize; N]>, Box<[Tensor1D<N>; CI]>),
) -> f64 {
    let (labels, inputs) = test_data;
    let outputs = chaos_network.forward_batch_no_grad(inputs);
    let outputs = head_network.forward_batch_no_grad(outputs);
    let guesses: Vec<usize> = (0..N)
        .map(|i| {
            let mut max: (usize, f64) = (0, outputs[0][i]);
            for ii in 0..HO {
                if outputs[ii][i] > max.1 {
                    max = (ii, outputs[ii][i]);
                }
            }
            max.0
        })
        .collect();
    let correct = guesses
        .iter()
        .enumerate()
        .filter(|(i, g)| **g == labels[*i])
        .count();
    (correct as f64) / N as f64
}

impl<const I: usize, const O: usize, const N: usize> StandardClassificationNetworkHandler<I, O, N> {
    pub fn train(&mut self) {
        let mut stdin = termion::async_stdin();
        let mut size_weight = 0.075;
        let mut current_chaos_network: ChaosNetwork<I, CHAOS_NETWORK_OUTPUTS, N> =
            ChaosNetwork::new(I, CHAOS_NETWORK_OUTPUTS);
        let mut current_head_network: HeadNetwork<CHAOS_NETWORK_OUTPUTS, O, N> =
            HeadNetwork::default();

        // Do the actual training
        for training_step in 0..self.max_training_steps {
            // Prep the data
            let batch_train_data: Vec<(Box<[usize; N]>, Box<[Tensor1D<N>; I]>)> = (0..self
                .steps_per_training_step)
                .map(|_i| {
                    let (train_labels, train_examples) = self.train_data.next();
                    (
                        train_labels,
                        transform_train_data_for_chaos_network(train_examples),
                    )
                })
                .collect();
            let batch_test_data: Vec<(Box<[usize; N]>, Box<[Tensor1D<N>; I]>)> = (0..self
                .validation_steps)
                .map(|_i| {
                    let (train_labels, train_examples) = self.test_data.next();
                    (
                        train_labels,
                        transform_train_data_for_chaos_network(train_examples),
                    )
                })
                .collect();

            // Do it!
            (current_chaos_network, current_head_network) = if training_step % 10 == 0 {
                let new_networks: Vec<(
                    ChaosNetwork<I, CHAOS_NETWORK_OUTPUTS, N>,
                    HeadNetwork<CHAOS_NETWORK_OUTPUTS, O, N>,
                    f64,
                )> = (0..MORPHS_PER_ITERATION)
                    .into_par_iter()
                    .map(|_i| {
                        let mut chaos_network = if training_step != 0 {
                            let mut network = current_chaos_network.clone();
                            let mut rng = rand::thread_rng();
                            let percent_nodes_to_add = rng.gen::<f64>() / 50.;
                            let percent_edges_to_add = rng.gen::<f64>() / 25.;
                            let percent_edges_to_remove = rng.gen::<f64>() / 10.;
                            // prune(&mut network, percent_edges_to_remove);
                            // grow(&mut network, percent_nodes_to_add, percent_edges_to_add);
                            network
                        } else {
                            ChaosNetwork::new(I, CHAOS_NETWORK_OUTPUTS)
                        };
                        let mut head_network = current_head_network.clone();
                        batch_train_data.iter().for_each(|batch| {
                            train_next_batch(&mut chaos_network, &mut head_network, batch)
                        });
                        let average_validation_accuracy: f64 = batch_test_data
                            .iter()
                            .map(|step| {
                                validate_next_batch(&mut chaos_network, &mut head_network, step)
                            })
                            .sum::<f64>()
                            / (self.validation_steps as f64);
                        (chaos_network, head_network, average_validation_accuracy)
                    })
                    .collect();

                // Sort the new networks
                let network_sizes: Vec<f64> = new_networks
                    .iter()
                    .map(|(n, _, _)| n.get_edge_count() as f64)
                    .collect();
                let min_network_size = network_sizes
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                let (new_chaos_network, new_head_network, average_validation_accuracy, score): (
                    ChaosNetwork<I, CHAOS_NETWORK_OUTPUTS, N>,
                    HeadNetwork<CHAOS_NETWORK_OUTPUTS, O, N>,
                    f64,
                    f64,
                ) = new_networks
                    .into_iter()
                    .map(|(n, h, ava)| {
                        let edge_count = n.get_edge_count() as f64;
                        (
                            n,
                            h,
                            ava,
                            ava + ((min_network_size / edge_count) * size_weight),
                        )
                    })
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                    .unwrap();
                println!(
                    "AVA: {} | Score: {}{:?}",
                    average_validation_accuracy, score, new_chaos_network
                );
                (new_chaos_network, new_head_network)
            } else {
                batch_train_data.iter().for_each(|batch| {
                    train_next_batch(&mut current_chaos_network, &mut current_head_network, batch)
                });
                (current_chaos_network, current_head_network)
            };

            // Match user input
            if let Some(line) = stdin.read_line().unwrap() {
                if line.len() > 0 {
                    let x = parse_input(line);
                    match x {
                        Ok((action, num1)) => match action {
                            Action::ChangeSizeWeight => {
                                size_weight = num1;
                                println!("\nUpdated Size Weight: {}\n", size_weight);
                            }
                            _ => (),
                        },
                        Err(error) => println!("ERROR: {:?}", error),
                    }
                }
            }
        }
    }
}
