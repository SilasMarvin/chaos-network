use crate::network::Network;
use crate::tensor_operations::Tensor1DNll;
use crate::tensors::Tensor1D;
use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;
use termion::input::TermRead;

const MORPHS_PER_ITERATION: usize = 1;

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

    fn next(&self, indexes: &[usize; N]) -> (Vec<usize>, Vec<Tensor1D<N>>) {
        let inputs: Vec<Tensor1D<N>> = (0..I)
            .map(|i| {
                let data: [f64; N] = indexes
                    .iter()
                    .map(|ii| self.data[*ii][i] as f64 / 255.)
                    .collect::<Vec<f64>>()
                    .try_into()
                    .unwrap();
                Tensor1D::new(data)
            })
            .collect();
        let labels: Vec<usize> = indexes
            .into_iter()
            .map(|i| self.labels[*i] as usize)
            .collect();
        (labels, inputs)
    }

    fn len(&self) -> usize {
        self.data.len()
    }
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

fn grow<const N: usize>(
    network: &mut Network<N>,
    percent_nodes_to_add: f64,
    percent_edges_to_add: f64,
) {
    let nodes_to_add = (network.nodes.len() as f64 * percent_nodes_to_add) as usize;
    let edges_to_add = (network.get_edge_count() as f64 * percent_edges_to_add) as usize;
    network.add_nodes(super::NodeKind::Normal, nodes_to_add);
    network.add_random_edges_for_random_nodes(edges_to_add);
}

fn prune<const N: usize>(network: &mut Network<N>, percent_edges_to_remove: f64) {
    let edges_to_remove = (network.get_edge_count() as f64 * percent_edges_to_remove) as usize;
    network.remove_weighted_edges(edges_to_remove);
}

fn train_next_batch<const N: usize>(
    network: &mut Network<N>,
    train_data: &(Vec<usize>, Vec<Tensor1D<N>>),
) {
    let (labels, inputs) = train_data;
    let outputs = network.forward_batch(inputs);
    let _loss = &mut Tensor1D::nll(outputs, labels, &mut network.tape);
    network.execute_and_apply_gradients();
}

fn validate_next_batch<const N: usize>(
    network: &mut Network<N>,
    test_data: &(Vec<usize>, Vec<Tensor1D<N>>),
) -> f64 {
    let (labels, inputs) = test_data;
    let outputs = network.forward_batch_no_grad(inputs);
    let guesses: Vec<usize> = (0..N)
        .map(|i| {
            let mut max: (usize, f64) = (0, outputs[0].data[i]);
            for ii in 0..network.leaves_count {
                if outputs[ii].data[i] > max.1 {
                    max = (ii, outputs[ii].data[i]);
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
        let mut current_network = Network::new(I, O);

        // Prep some stuff for getting random data points
        let train_data_distribution = Uniform::from(0..self.train_data.len());
        let test_data_distribution = Uniform::from(0..self.test_data.len());
        let mut rng = thread_rng();

        // Do the actual training
        for training_step in 0..self.max_training_steps {
            // Prep the random indexes
            let batch_train_data = (0..self.steps_per_training_step)
                .map(|_i| {
                    self.train_data.next(
                        &(0..N)
                            .map(|_ii| train_data_distribution.sample(&mut rng))
                            .collect::<Vec<usize>>()
                            .try_into()
                            .unwrap(),
                    )
                })
                .collect::<Vec<(Vec<usize>, Vec<Tensor1D<N>>)>>();
            let batch_test_data = (0..self.validation_steps)
                .map(|_i| {
                    self.train_data.next(
                        &(0..N)
                            .map(|_ii| test_data_distribution.sample(&mut rng))
                            .collect::<Vec<usize>>()
                            .try_into()
                            .unwrap(),
                    )
                })
                .collect::<Vec<(Vec<usize>, Vec<Tensor1D<N>>)>>();

            // Do it!
            current_network = if training_step % 10 == 0 {
                let new_networks: Vec<(Network<N>, f64)> = (0..MORPHS_PER_ITERATION)
                    .into_par_iter()
                    .map(|_i| {
                        let mut network = if training_step != 0 {
                            let mut network = current_network.clone();
                            let mut rng = rand::thread_rng();
                            let percent_nodes_to_add = rng.gen::<f64>() / 50.;
                            let percent_edges_to_add = rng.gen::<f64>() / 25.;
                            let percent_edges_to_remove = rng.gen::<f64>() / 10.;
                            prune(&mut network, percent_edges_to_remove);
                            grow(&mut network, percent_nodes_to_add, percent_edges_to_add);
                            network
                        } else {
                            Network::new(I, O)
                        };
                        batch_train_data
                            .iter()
                            .for_each(|batch| train_next_batch(&mut network, batch));
                        let average_validation_accuracy: f64 = (0..self.validation_steps)
                            .map(|step| validate_next_batch(&mut network, &batch_test_data[step]))
                            .sum::<f64>()
                            / (self.validation_steps as f64);
                        (network, average_validation_accuracy)
                    })
                    .collect();

                // Sort the new networks
                let network_sizes: Vec<f64> = new_networks
                    .iter()
                    .map(|(n, _)| n.get_edge_count() as f64)
                    .collect();
                let min_network_size = network_sizes
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                let (new_network, average_validation_accuracy, score): (Network<N>, f64, f64) =
                    new_networks
                        .into_iter()
                        .map(|(n, ava)| {
                            let edge_count = n.get_edge_count() as f64;
                            (
                                n,
                                ava,
                                ava + ((min_network_size / edge_count) * size_weight),
                            )
                        })
                        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                        .unwrap();
                println!(
                    "AVA: {} | Score: {}{:?}",
                    average_validation_accuracy, score, new_network
                );
                new_network
            } else {
                batch_train_data
                    .iter()
                    .for_each(|batch| train_next_batch(&mut current_network, batch));
                current_network
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
