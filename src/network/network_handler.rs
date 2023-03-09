use crate::network::Network;
use crate::tensors::{Tensor, Tensor1D};
use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;

const POPULATION_SIZE: usize = 64;

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
                Tensor1D::new_without_tape(data)
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
    percent_connections_to_add: f64,
) {
    let nodes_to_add = (network.nodes.len() as f64 * percent_nodes_to_add) as usize;
    let connections_to_add =
        (network.get_connection_count() as f64 * percent_connections_to_add) as usize;
    network.add_nodes(super::NodeKind::Normal, nodes_to_add);
    network.add_random_connections(connections_to_add);
}

fn prune<const N: usize>(network: &mut Network<N>, percent_connections_to_remove: f64) {
    let connections_to_remove =
        (network.get_connection_count() as f64 * percent_connections_to_remove) as usize;
    network.remove_weighted_connections(connections_to_remove);
    network.prune_unconnected_nodes();
}

fn train_next_batch<const N: usize>(
    network: &mut Network<N>,
    train_data: &(Vec<usize>, Vec<Tensor1D<N>>),
) {
    let (labels, inputs) = train_data;
    let outputs = network.forward_batch(inputs);
    let loss = &mut Tensor1D::nll(outputs, labels);
    let grads = loss.backward();
    network.apply_gradients(grads);
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
        // Create the initial population
        let mut population: Vec<Network<N>> =
            (0..POPULATION_SIZE).map(|_i| Network::new(I, O)).collect();

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
            // Current population and maybe new networks
            population = if training_step != 0 && training_step % 10 == 0 {
                let new_networks = population.iter().map(|x| x.clone()).collect();
                let new_morphed_networks =
                    self.train_population(new_networks, &batch_train_data, &batch_test_data, true);
                println!(
                    "New Morphed Networks: {:?}",
                    new_morphed_networks
                        .iter()
                        .take(5)
                        .map(|(n, ava, score)| (*ava, *score, n.get_connection_count()))
                        .collect::<Vec<(f64, f64, i32)>>()
                );
                let new_population_networks =
                    self.train_population(population, &batch_train_data, &batch_test_data, false);
                println!(
                    "New Population Networks: {:?}",
                    new_population_networks
                        .iter()
                        .take(5)
                        .map(|(n, ava, score)| (*ava, *score, n.get_connection_count()))
                        .collect::<Vec<(f64, f64, i32)>>()
                );

                // Merge them together
                new_population_networks
                    .into_iter()
                    .zip(new_morphed_networks.into_iter())
                    .rev()
                    .skip(POPULATION_SIZE / 2)
                    .map(|(a, b)| vec![a.0, b.0])
                    .flatten()
                    .collect()
            } else {
                self.train_population(population, &batch_train_data, &batch_test_data, false)
                    .into_iter()
                    // .inspect(|(_, ava, _)| println!("{}", ava))
                    .map(|(n, _, _)| n)
                    .collect()
            }
        }
    }

    fn train_population(
        &self,
        population: Vec<Network<N>>,
        batch_train_data: &Vec<(Vec<usize>, Vec<Tensor1D<N>>)>,
        batch_test_data: &Vec<(Vec<usize>, Vec<Tensor1D<N>>)>,
        do_morph: bool,
    ) -> Vec<(Network<N>, f64, f64)> {
        // Do the training and validation
        let new_networks: Vec<(Network<N>, f64)> = population
            .into_par_iter()
            .map(|mut network| {
                if do_morph {
                    let mut rng = rand::thread_rng();
                    let percent_nodes_to_add = rng.gen::<f64>() / 50.;
                    let percent_connections_to_add = rng.gen::<f64>() / 50.;
                    let percent_connections_to_remove = rng.gen::<f64>() / 25.;
                    grow(
                        &mut network,
                        percent_nodes_to_add,
                        percent_connections_to_add,
                    );
                    prune(&mut network, percent_connections_to_remove);
                }
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
        let network_sizes: Vec<f64> = new_networks
            .iter()
            .map(|(n, _)| n.get_connection_count() as f64)
            .collect();
        let min_network_size = network_sizes
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let mut new_networks: Vec<(Network<N>, f64, f64)> = new_networks
            .into_iter()
            .map(|(n, ava)| {
                let connection_count = n.get_connection_count() as f64;
                (n, ava, ava + ((min_network_size / connection_count) * 0.05))
            })
            .collect();
        new_networks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        new_networks
    }
}
