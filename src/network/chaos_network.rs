use rand::distributions::{Uniform, WeightedIndex};
use rand::prelude::*;
use rand::Rng;

use rustc_hash::{FxHashMap, FxHashSet};
use std::boxed::Box;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use crate::gradients::Gradients;
use crate::gradients::Tape;
use crate::network::optimizers::{AdamOptimizer, Optimizer};
use crate::tensor_operations::{Tensor0DMul, Tensor1DAdd, Tensor1DMish, Tensor1DSplitOnAdd};
use crate::tensors::{Tensor0D, Tensor1D, WithTape, WithoutTape};

pub static NODE_COUNT: AtomicI32 = AtomicI32::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Input,
    Normal,
    Leaf,
}

pub enum ShiftDirection {
    Forward,
    Backward,
}

#[derive(Default)]
pub struct Network<const N: usize> {
    pub inputs_count: usize,
    pub leaves_count: usize,
    pub nodes: Vec<Node<N>>,
    pub connections_to: FxHashMap<i32, Vec<usize>>,
    pub tape: Tape<N>,
}

#[derive(Clone)]
pub struct Node<const N: usize> {
    pub id: i32,
    pub weights: Vec<Tensor0D<N, WithTape>>,
    pub kind: NodeKind,
    pub optimizer: Box<dyn Optimizer>,
}

impl<const N: usize> Network<N> {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let mut network: Network<N> = Network::default();
        network.add_nodes(NodeKind::Leaf, outputs);
        network.add_nodes(NodeKind::Input, inputs);
        network
    }

    pub fn add_nodes(&mut self, kind: NodeKind, count: usize) {
        match kind {
            NodeKind::Normal => {
                let node_index = self.batch_insert_normal_nodes(count);
                for i in 0..(count) {
                    for _ii in 0..10 {
                        self.add_node_connection_to(node_index + i);
                        self.add_node_connection_from(node_index + i);
                    }
                }
            }
            NodeKind::Input => {
                self.inputs_count += count;
                let mut nodes: Vec<Node<N>> = (0..count)
                    .map(|_i| Node::new(kind, &mut self.tape))
                    .collect();
                for _i in 0..count {
                    self.nodes.insert(0, nodes.remove(0));
                }
                self.shift_all_connections_after(0, count, ShiftDirection::Forward);
                if self.leaves_count == 0 {
                    panic!("This should be called after leaves are added for now")
                }
                let distribution_between =
                    Uniform::from(self.nodes.len() - self.leaves_count..self.nodes.len());
                let mut rng = rand::thread_rng();
                let odds_of_being_picked = (count as f64 * 0.1) / (count as f64);
                for i in 0..1000 {
                    if odds_of_being_picked > rng.gen::<f64>() {
                        let input_node_index = distribution_between.sample(&mut rng);
                        self.add_connection_between(i, input_node_index);
                    }
                }
            }
            NodeKind::Leaf => {
                self.leaves_count += count;
                for _i in 0..count {
                    self.nodes.push(Node::new(kind, &mut self.tape));
                }
            }
        }
    }

    fn batch_insert_normal_nodes(&mut self, count: usize) -> usize {
        let node_index = if self.nodes.len() - self.inputs_count - self.leaves_count > 0 {
            let mut rng = rand::thread_rng();
            let node_index =
                rng.gen_range(0..(self.nodes.len() - self.inputs_count - self.leaves_count));
            node_index + self.inputs_count
        } else {
            self.inputs_count
        };
        for _i in 0..count {
            self.nodes
                .insert(node_index, Node::new(NodeKind::Normal, &mut self.tape));
        }
        self.shift_all_connections_after(node_index, count, ShiftDirection::Forward);
        node_index
    }

    pub fn shift_all_connections_after(
        &mut self,
        after: usize,
        count: usize,
        direction: ShiftDirection,
    ) {
        for (_key, value) in self.connections_to.iter_mut() {
            value.iter_mut().for_each(|u| {
                if *u >= after {
                    match direction {
                        ShiftDirection::Forward => *u += count,
                        ShiftDirection::Backward => *u -= count,
                    }
                }
            });
        }
    }

    pub fn remove_all_connections_to(&mut self, node_index: usize) {
        for i in 0..self.nodes.len() - self.leaves_count {
            let connections = self.connections_to.get_mut(&self.nodes[i].id).unwrap();
            let mut removal_adjust = 0;
            for ii in 0..connections.len() {
                if connections[ii - removal_adjust] == node_index {
                    connections.remove(ii - removal_adjust);
                    // Remove the corresponding weight, adjust for the prescence of bias if it is a
                    // normal node
                    if self.nodes[i].kind == NodeKind::Normal {
                        self.nodes[i].weights.remove(ii + 1 - removal_adjust);
                    } else {
                        self.nodes[i].weights.remove(ii - removal_adjust);
                    }
                    removal_adjust += 1;
                }
            }
        }
    }

    fn add_connection_between(&mut self, node_index: usize, node2_index: usize) {
        match self.connections_to.get_mut(&self.nodes[node_index].id) {
            Some(connections) => connections.push(node2_index),
            None => {
                self.connections_to
                    .insert(self.nodes[node_index].id, vec![node2_index]);
            }
        };
        self.nodes[node_index].add_weight(&mut self.tape);
    }

    fn add_node_connection_to(&mut self, node_index: usize) {
        let mut rng = rand::thread_rng();
        let node2_index = match self.connections_to.get(&self.nodes[node_index].id) {
            Some(connections) => {
                let mut lock_break = 0;
                loop {
                    let index =
                        rng.gen_range((node_index + 1).max(self.inputs_count)..self.nodes.len());
                    if !connections.contains(&index) {
                        break index;
                    }
                    if lock_break > 1000 {
                        return;
                    }
                    lock_break += 1;
                }
            }
            None => rng.gen_range(node_index + 1..self.nodes.len()),
        };
        self.add_connection_between(node_index, node2_index);
    }

    fn add_node_connection_from(&mut self, node_index: usize) {
        let mut rng = rand::thread_rng();
        let mut lock_break = 0;
        let node2_index = loop {
            let index = rng.gen_range(0..node_index);
            match self.connections_to.get(&self.nodes[index].id) {
                Some(connections) => {
                    if !connections.contains(&node_index) {
                        break index;
                    }
                }
                None => break index,
            }
            if lock_break > 1000 {
                return;
            }
            lock_break += 1;
        };
        self.add_connection_between(node2_index, node_index);
    }

    pub fn forward_batch(&mut self, input: &Vec<Tensor1D<N>>) -> Vec<Tensor1D<N, WithTape>> {
        let mut output: Vec<Tensor1D<N, WithTape>> = Vec::new();
        output.resize(self.leaves_count, Tensor1D::new([0.; N]));
        let mut running_values: Vec<Option<Tensor1D<N, WithTape>>> = Vec::new();
        running_values.resize(self.nodes.len(), None);
        let nodes_len = self.nodes.len();
        for (i, node) in self.nodes.iter_mut().enumerate() {
            match node.kind {
                NodeKind::Input => {
                    if let Some(connections) = self.connections_to.get(&node.id) {
                        if connections.is_empty() {
                            continue;
                        }
                        for (ii, connection) in connections.iter().enumerate() {
                            let mut x =
                                node.weights[ii].mul_left_by_reference(&input[i], &mut self.tape);
                            let running_value = &mut running_values[*connection];
                            running_values[*connection] = match running_value {
                                Some(rv) => Some(x.add(rv, &mut self.tape)),
                                None => Some(x),
                            }
                        }
                    }
                }
                NodeKind::Normal => {
                    let connections = self.connections_to.get(&node.id).unwrap();
                    let running_value = &mut running_values[i];
                    let mut go_in = match running_value.as_mut() {
                        Some(mut rv) => {
                            let mut bias = node.weights[0].mul_left_by_reference(
                                &mut Tensor1D::<N, WithoutTape>::new([1.; N]),
                                &mut self.tape,
                            );
                            bias.add(&mut rv, &mut self.tape)
                        }
                        None => Tensor0DMul::mul(
                            &mut node.weights[i],
                            &mut Tensor1D::<N, WithoutTape>::new([1.; N]),
                            &mut self.tape,
                        ),
                    };
                    let go_in = go_in.mish(&mut self.tape);
                    let mut go_in = match connections.len() > 1 {
                        true => go_in.split_on_add(connections.len(), &mut self.tape),
                        _ => vec![go_in],
                    };
                    for (ii, connection) in connections.iter().enumerate() {
                        let mut x = Tensor0DMul::mul(
                            &mut node.weights[ii + 1],
                            &mut go_in.pop().unwrap(),
                            &mut self.tape,
                        );
                        let running_value = &mut running_values[*connection];
                        running_values[*connection] = match running_value {
                            Some(rv) => Some(x.add(rv, &mut self.tape)),
                            None => Some(x),
                        }
                    }
                }
                NodeKind::Leaf => {
                    let val = std::mem::replace(&mut running_values[i], None);
                    output[nodes_len - i - 1] = val.unwrap();
                }
            }
        }
        output
    }

    pub fn forward_batch_no_grad(
        &mut self,
        input: &Vec<Tensor1D<N>>,
    ) -> Vec<Tensor1D<N, WithoutTape>> {
        let mut output: Vec<Tensor1D<N, WithoutTape>> = Vec::new();
        output.resize(self.leaves_count, Tensor1D::new([0.; N]));
        let mut running_values: Vec<Option<Tensor1D<N, WithoutTape>>> = Vec::new();
        running_values.resize(self.nodes.len(), None);
        let nodes_len = self.nodes.len();
        for (i, node) in self.nodes.iter_mut().enumerate() {
            match node.kind {
                NodeKind::Input => {
                    if let Some(connections) = self.connections_to.get(&node.id) {
                        if connections.is_empty() {
                            continue;
                        }
                        for (ii, connection) in connections.iter().enumerate() {
                            let mut x =
                                node.weights[ii].mul_explicit_no_grad(&input[i], &mut self.tape);
                            let running_value = &mut running_values[*connection];
                            running_values[*connection] = match running_value {
                                Some(rv) => Some(x.add(rv, &mut self.tape).to_without_tape()),
                                None => Some(x.to_without_tape()),
                            }
                        }
                    }
                }
                NodeKind::Normal => {
                    let connections = self.connections_to.get(&node.id).unwrap();
                    let running_value = &mut running_values[i];
                    let mut go_in = match running_value.as_mut() {
                        Some(mut rv) => {
                            let mut bias = node.weights[0].mul_explicit_no_grad(
                                &mut Tensor1D::<N, WithoutTape>::new([1.; N]),
                                &mut self.tape,
                            );
                            bias.add(&mut rv, &mut self.tape)
                        }
                        None => node.weights[i].mul_explicit_no_grad(
                            &mut Tensor1D::<N, WithoutTape>::new([1.; N]),
                            &mut self.tape,
                        ),
                    };
                    let go_in = go_in.mish(&mut self.tape);
                    for (ii, connection) in connections.iter().enumerate() {
                        let mut x =
                            node.weights[ii + 1].mul_explicit_no_grad(&go_in, &mut self.tape);
                        let running_value = &mut running_values[*connection];
                        running_values[*connection] = match running_value {
                            Some(rv) => Some(x.add(rv, &mut self.tape).to_without_tape()),
                            None => Some(x.to_without_tape()),
                        }
                    }
                }
                NodeKind::Leaf => {
                    let val = std::mem::replace(&mut running_values[i], None);
                    output[nodes_len - i - 1] = val.unwrap();
                }
            }
        }
        output
    }

    pub fn add_random_connections(&mut self, count: usize) {
        let mut rng = rand::thread_rng();
        for _i in 0..count {
            let node_index = rng.gen_range(0..self.nodes.len() - self.leaves_count);
            self.add_node_connection_to(node_index);
        }
    }

    pub fn remove_weighted_connections(&mut self, count: usize) {
        let mut rng = rand::thread_rng();
        let weights = self
            .nodes
            .iter()
            .enumerate()
            // This filters out inputs which we don't remove even if they have 0 connections
            // Also filter out leaves
            .filter(|(_i, n)| n.weights.len() > 0)
            .map(|(i, n)| {
                // If it is a normal node, skip over the bias weight
                match self.nodes[i].kind {
                    NodeKind::Normal => n
                        .weights
                        .iter()
                        .enumerate()
                        .skip(1)
                        .map(move |(ii, w)| (i, ii, w.data))
                        .collect::<Vec<(usize, usize, f64)>>(),
                    _ => n
                        .weights
                        .iter()
                        .enumerate()
                        .map(move |(ii, w)| (i, ii, w.data))
                        .collect::<Vec<(usize, usize, f64)>>(),
                }
            })
            .flatten()
            .collect::<Vec<(usize, usize, f64)>>();
        if weights.is_empty() {
            return;
        }
        let weighted = WeightedIndex::new(weights.iter().map(|w| w.2.powi(2) + 1.0)).unwrap();
        let mut sampled = FxHashSet::default();
        for _i in 0..count {
            let item = weights[weighted.sample(&mut rng)];
            if sampled.contains(&item.0) {
                continue;
            }
            sampled.insert(item.0);
            let removal_index = match self.nodes[item.0].kind {
                NodeKind::Normal => item.1 - 1,
                _ => item.1,
            };
            // NOTE Removing done here
            self.connections_to
                .get_mut(&self.nodes[item.0].id)
                .unwrap()
                .remove(removal_index);
            // self.nodes[item.0].weights.remove(item.1);
            self.nodes[item.0].remove_weight(item.1, &mut self.tape);
        }
    }

    pub fn prune_unconnected_nodes(&mut self) {
        let mut removal_adjust = 0;
        for i in self.inputs_count..(self.nodes.len() - self.leaves_count) {
            let real_i = i - removal_adjust;
            if self
                .connections_to
                .get(&self.nodes[real_i].id)
                .unwrap()
                .len()
                == 0
            {
                self.connections_to.remove(&self.nodes[real_i].id);
                self.nodes.remove(real_i);
                self.remove_all_connections_to(real_i);
                self.shift_all_connections_after(real_i, 1, ShiftDirection::Backward);
                removal_adjust += 1;
            }
        }
    }

    pub fn execute_and_apply_gradients(&mut self) {
        let mut grads = self.tape.execute();
        for n in self.nodes.iter_mut() {
            n.apply_gradients(&mut grads);
        }
    }

    pub fn get_connection_count(&self) -> i32 {
        self.connections_to
            .iter()
            .fold(0, |acc, (_key, value)| acc + value.len() as i32)
    }
}

impl<const N: usize> Node<N> {
    fn new(kind: NodeKind, tape: &mut Tape<N>) -> Self {
        match kind {
            NodeKind::Normal => {
                let mut node = Self {
                    id: NODE_COUNT.fetch_add(1, Ordering::SeqCst),
                    weights: Vec::new(),
                    kind,
                    optimizer: Box::new(AdamOptimizer::default()),
                };
                node.add_weight(tape);
                node
            }
            NodeKind::Input => Self {
                id: NODE_COUNT.fetch_add(1, Ordering::SeqCst),
                weights: Vec::new(),
                kind,
                optimizer: Box::new(AdamOptimizer::default()),
            },
            NodeKind::Leaf => Self {
                id: NODE_COUNT.fetch_add(1, Ordering::SeqCst),
                weights: Vec::new(),
                kind,
                optimizer: Box::new(AdamOptimizer::default()),
            },
        }
    }

    fn add_weight(&mut self, tape: &mut Tape<N>) {
        let mut rng = rand::thread_rng();
        let w = (rng.gen::<f64>() - 0.5) / 100.;
        let mut v: Tensor0D<N, WithTape> = Tensor0D::new(w);
        v.set_id_grad_for(tape.get_next_network_tensor_id());
        self.weights.push(v);
    }

    fn remove_weight(&mut self, index: usize, tape: &mut Tape<N>) {
        tape.remove_network_tensor(self.weights[index].id);
        self.weights.remove(index);
    }

    fn apply_gradients(&mut self, gradients: &mut Gradients<N>) {
        for w in self.weights.iter_mut() {
            let w_gradients = gradients.remove_or_0(w.id);
            let averaged_gradients: f64 = w_gradients.data.iter().sum::<f64>() / (N as f64);
            if averaged_gradients != 0. {
                // w.data -= 0.01 * averaged_gradients;
                w.data -= self.optimizer.update(averaged_gradients);
            }
        }
    }
}
