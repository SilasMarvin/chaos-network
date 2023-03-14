use rand::distributions::{Uniform, WeightedIndex};
use rand::prelude::*;
use rand::Rng;

use rustc_hash::{FxHashMap, FxHashSet};
use std::boxed::Box;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use crate::gradients::Gradients;
use crate::gradients::Tape;
use crate::network::optimizers::{AdamOptimizer, Optimizer};
use crate::tensor_operations::{Tensor0DMul, Tensor1DAdd, Tensor1DMish, Tensor1DSplitOnAdd};
use crate::tensors::{Tensor0D, Tensor1D, WithTape, WithoutTape};

pub static NODE_COUNT: AtomicUsize = AtomicUsize::new(0);

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
    pub edges: HashMap<usize, Vec<usize>>,
    pub edges_to_count: HashMap<usize, usize>,
    pub tape: Tape<N>,
}

#[derive(Clone)]
pub struct Node<const N: usize> {
    pub id: usize,
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
                let mut rng = rand::thread_rng();
                for _i in 0..count {
                    let node_index = if self.nodes.len() > self.inputs_count + self.leaves_count {
                        rng.gen_range(self.inputs_count..(self.nodes.len() - self.leaves_count))
                    } else {
                        self.inputs_count
                    };
                    self.nodes
                        .insert(node_index, Node::new(NodeKind::Normal, &mut self.tape));
                    self.shift_all_edges_after(node_index, 1, ShiftDirection::Forward);
                    self.add_random_edge(node_index);
                    self.add_random_edge_to(node_index);
                }
            }
            NodeKind::Input => {
                if self.leaves_count == 0 {
                    panic!("This should be called after leaves are added for now")
                }
                self.inputs_count += count;
                for _i in 0..count {
                    self.nodes
                        .insert(0, Node::new(NodeKind::Input, &mut self.tape));
                }
                self.shift_all_edges_after(0, count, ShiftDirection::Forward);
                let mut rng = rand::thread_rng();
                let odds_of_being_picked = (count as f64 * 0.1) / (count as f64);
                for i in 0..count {
                    if odds_of_being_picked > rng.gen::<f64>() {
                        self.add_random_edge(i);
                    }
                }
            }
            NodeKind::Leaf => {
                self.leaves_count += count;
                for _i in 0..count {
                    self.nodes.push(Node::new(NodeKind::Leaf, &mut self.tape));
                }
            }
        }
    }

    pub fn shift_all_edges_after(&mut self, after: usize, count: usize, direction: ShiftDirection) {
        for (_key, value) in self.edges.iter_mut() {
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

    // pub fn remove_all_edges_to(&mut self, node_index: usize) {
    //     for i in 0..self.nodes.len() - self.leaves_count {
    //         if let Some(edges) = self.edges.get_mut(&self.nodes[i].id) {
    //             for ii in 0..edges.len() {
    //                 if edges[ii] == node_index {
    //                     edges.remove(ii);
    //                     if self.nodes[i].kind == NodeKind::Normal {
    //                         self.nodes[i].remove_weight(ii + 1, &mut self.tape);
    //                     } else {
    //                         self.nodes[i].remove_weight(ii, &mut self.tape);
    //                     }
    //                     break;
    //                 }
    //             }
    //         }
    //     }
    //     self.edges_to_count.remove(&self.nodes[node_index].id);
    // }

    fn add_edge_between(&mut self, node_index: usize, node2_index: usize) {
        match self.edges.get_mut(&self.nodes[node_index].id) {
            Some(edges) => edges.push(node2_index),
            None => {
                self.edges
                    .insert(self.nodes[node_index].id, vec![node2_index]);
            }
        };
        match self.edges_to_count.get_mut(&self.nodes[node2_index].id) {
            Some(edge_count) => *edge_count += 1,
            None => {
                self.edges_to_count.insert(self.nodes[node2_index].id, 1);
            }
        };
        self.nodes[node_index].add_weight(&mut self.tape);
    }

    fn add_random_edge(&mut self, node_index: usize) {
        let mut rng = rand::thread_rng();
        let node2_index = match self.edges.get(&self.nodes[node_index].id) {
            Some(edges) => {
                let mut lock_break = 0;
                loop {
                    let index =
                        rng.gen_range((node_index + 1).max(self.inputs_count)..self.nodes.len());
                    if !edges.contains(&index) {
                        break index;
                    }
                    if lock_break > 1000 {
                        println!("LOCK BREAKING FOR ADDING RANDOM EDGE");
                        return;
                    }
                    lock_break += 1;
                }
            }
            None => rng.gen_range((node_index + 1).max(self.inputs_count)..self.nodes.len()),
        };
        self.add_edge_between(node_index, node2_index);
    }

    fn add_random_edge_to(&mut self, node_index: usize) {
        let mut rng = rand::thread_rng();
        let mut lock_break = 0;
        let node2_index = loop {
            let index = rng.gen_range(0..node_index);
            match self.edges.get(&self.nodes[index].id) {
                Some(edges) => {
                    if !edges.contains(&node_index) {
                        break index;
                    }
                }
                None => break index,
            }
            if lock_break > 1000 {
                println!("LOCK BREAKING FOR ADDING RANDOM EDGE TO");
                return;
            }
            lock_break += 1;
        };
        self.add_edge_between(node2_index, node_index);
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
                    if let Some(edges) = self.edges.get(&node.id) {
                        if edges.is_empty() {
                            continue;
                        }
                        for (ii, edge) in edges.iter().enumerate() {
                            let mut x =
                                node.weights[ii].mul_left_by_reference(&input[i], &mut self.tape);
                            let running_value = &mut running_values[*edge];
                            running_values[*edge] = match running_value {
                                Some(rv) => Some(x.add(rv, &mut self.tape)),
                                None => Some(x),
                            }
                        }
                    }
                }
                NodeKind::Normal => {
                    let edges = self.edges.get(&node.id).unwrap();
                    let running_value = &mut running_values[i];
                    let mut go_in = match running_value.as_mut() {
                        Some(mut rv) => {
                            let mut bias = node.weights[0].mul_left_by_reference(
                                &Tensor1D::<N, WithoutTape>::new([1.; N]),
                                &mut self.tape,
                            );
                            bias.add(&mut rv, &mut self.tape)
                        }
                        None => {
                            println!("{:?}", self.nodes[i]);
                            println!("Edges: {:?}", self.edges.get(&self.nodes[i].id));
                            println!(
                                "Edges To Count: {:?}",
                                self.edges_to_count.get(&self.nodes[i].id)
                            );
                            println!("Node Index: {}", i);
                            for (key, value) in self.edges.iter() {
                                for u in value.iter() {
                                    if *u == i {
                                        println!("WE FOUND A CONNECTION");
                                    }
                                }
                            }
                            panic!("We should not be at a normal node that does not have a running value");
                        }
                    };
                    let go_in = go_in.mish(&mut self.tape);
                    let mut go_in = match edges.len() > 1 {
                        true => go_in.split_on_add(edges.len(), &mut self.tape),
                        _ => vec![go_in],
                    };
                    for (ii, edge) in edges.iter().enumerate() {
                        let mut x = node.weights[ii + 1].mul(&mut go_in[ii], &mut self.tape);
                        let running_value = &mut running_values[*edge];
                        running_values[*edge] = match running_value {
                            Some(rv) => Some(x.add(rv, &mut self.tape)),
                            None => Some(x),
                        }
                    }
                }
                NodeKind::Leaf => {
                    // NOTE: This can panic as not every leaf is guaranteed a edge
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
                    if let Some(edges) = self.edges.get(&node.id) {
                        if edges.is_empty() {
                            continue;
                        }
                        for (ii, edge) in edges.iter().enumerate() {
                            let mut x =
                                node.weights[ii].mul_explicit_no_grad(&input[i], &mut self.tape);
                            let running_value = &mut running_values[*edge];
                            running_values[*edge] = match running_value {
                                Some(rv) => Some(x.add(rv, &mut self.tape).to_without_tape()),
                                None => Some(x.to_without_tape()),
                            }
                        }
                    }
                }
                NodeKind::Normal => {
                    let edges = self.edges.get(&node.id).unwrap();
                    let running_value = &mut running_values[i];
                    let mut go_in = match running_value.as_mut() {
                        Some(mut rv) => {
                            let mut bias = node.weights[0].mul_explicit_no_grad(
                                &mut Tensor1D::<N, WithoutTape>::new([1.; N]),
                                &mut self.tape,
                            );
                            bias.add(&mut rv, &mut self.tape)
                        }
                        None => panic!(
                            "We should not be at a normal node that does not have a running value"
                        ),
                    };
                    let go_in = go_in.mish(&mut self.tape);
                    for (ii, edge) in edges.iter().enumerate() {
                        let mut x =
                            node.weights[ii + 1].mul_explicit_no_grad(&go_in, &mut self.tape);
                        let running_value = &mut running_values[*edge];
                        running_values[*edge] = match running_value {
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

    pub fn add_random_edges_for_random_nodes(&mut self, count: usize) {
        let mut rng = rand::thread_rng();
        for _i in 0..count {
            let node_index = rng.gen_range(0..self.nodes.len() - self.leaves_count);
            self.add_random_edge(node_index);
        }
    }

    pub fn remove_weighted_edges(&mut self, count: usize) {
        // let mut rng = rand::thread_rng();
        // let weights = self
        //     .nodes
        //     .iter()
        //     .enumerate()
        //     // This filters out inputs which we don't remove even if they have 0 edges
        //     // Also filter out leaves
        //     .filter(|(_i, n)| n.weights.len() > 0)
        //     .map(|(i, n)| {
        //         // If it is a normal node, skip over the bias weight
        //         match self.nodes[i].kind {
        //             NodeKind::Normal => n
        //                 .weights
        //                 .iter()
        //                 .enumerate()
        //                 .skip(1)
        //                 .map(move |(ii, w)| (i, ii, w.data))
        //                 .collect::<Vec<(usize, usize, f64)>>(),
        //             _ => n
        //                 .weights
        //                 .iter()
        //                 .enumerate()
        //                 .map(move |(ii, w)| (i, ii, w.data))
        //                 .collect::<Vec<(usize, usize, f64)>>(),
        //         }
        //     })
        //     .flatten()
        //     .collect::<Vec<(usize, usize, f64)>>();
        // if weights.is_empty() {
        //     return;
        // }
        // let weighted = WeightedIndex::new(weights.iter().map(|w| w.2.powi(2) + 1.0)).unwrap();
        // let mut sampled = FxHashSet::default();
        // for _i in 0..count {
        //     let item = weights[weighted.sample(&mut rng)];
        //     if sampled.contains(&item.0) {
        //         continue;
        //     }
        //     sampled.insert(item.0);
        //     let edge_removal_index = match self.nodes[item.0].kind {
        //         NodeKind::Normal => item.1 - 1,
        //         _ => item.1,
        //     };
        //     let removed_edge_node_index = self
        //         .edges
        //         .get_mut(&self.nodes[item.0].id)
        //         .unwrap()
        //         .remove(edge_removal_index);
        //     println!("{}", removed_edge_node_index);
        //     *self
        //         .edges_to_count
        //         .get_mut(&self.nodes[removed_edge_node_index].id)
        //         .unwrap() -= 1;
        //     self.nodes[item.0].remove_weight(item.1, &mut self.tape);
        // }

        let mut rng = thread_rng();
        for _i in 0..count {
            let node_index = rng.gen_range(0..self.nodes.len() - self.leaves_count);
            if let Some(edges) = self.edges.get_mut(&self.nodes[node_index].id) {
                if edges.len() < 2 {
                    continue;
                }
                let edge_index = rng.gen_range(0..edges.len());
                let removed_node_index = edges.remove(edge_index);
                println!(
                    "Removed connection from {} -> {}",
                    node_index, removed_node_index
                );
                // Adjust weight removal index in the case of a bias
                let weight_removal_index = match self.nodes[node_index].kind {
                    NodeKind::Normal => edge_index + 1,
                    _ => edge_index,
                };
                self.nodes[node_index].remove_weight(weight_removal_index, &mut self.tape);
                // Handle the removed edge node
                let removed_node_edge_count = self
                    .edges_to_count
                    .get_mut(&self.nodes[removed_node_index].id)
                    .unwrap();
                *removed_node_edge_count -= 1;
                if self.nodes[removed_node_index].kind == NodeKind::Normal {
                    if *removed_node_edge_count == 0 {
                        self.remove_node_with_no_edges_to_it(removed_node_index);
                    }
                }
            }
        }
    }

    pub fn remove_node_with_no_edges_to_it(&mut self, node_index: usize) {
        println!("Removing: {}", node_index);
        let edges = self.edges.remove(&self.nodes[node_index].id).unwrap();
        self.nodes[node_index].remove_all_weights(&mut self.tape);
        self.nodes.remove(node_index);
        self.shift_all_edges_after(node_index, 1, ShiftDirection::Backward);
        for edge_node_index in edges.into_iter() {
            let edge_node_edge_count = self
                .edges_to_count
                .get_mut(&self.nodes[edge_node_index - 1].id) // We must shift manually
                .unwrap();
            *edge_node_edge_count -= 1;
            if self.nodes[edge_node_index].kind == NodeKind::Normal {
                if *edge_node_edge_count == 0 {
                    println!("Recursing");
                    self.remove_node_with_no_edges_to_it(edge_node_index);
                }
            }
        }
    }

    pub fn execute_and_apply_gradients(&mut self) {
        let mut grads = self.tape.execute();
        for n in self.nodes.iter_mut() {
            n.apply_gradients(&mut grads);
        }
    }

    pub fn get_edge_count(&self) -> i32 {
        self.edges
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

    fn remove_all_weights(&mut self, tape: &mut Tape<N>) {
        while self.weights.len() > 0 {
            self.remove_weight(0, tape);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_nodes() {
        let mut network: Network<10> = Network::new(10, 10);
        assert_eq!(network.inputs_count, 10);
        assert_eq!(network.leaves_count, 10);
        assert_eq!(network.nodes.len(), 20);
        network.add_nodes(NodeKind::Normal, 10);
        assert_eq!(network.nodes.len(), 30);
    }

    #[test]
    fn shift_all_edges_after() {
        let mut network: Network<10> = Network::new(10, 10);
        let edges_before = network.edges.clone();
        network.shift_all_edges_after(5, 2, ShiftDirection::Forward);
        for (key, eb) in edges_before.iter() {
            let ea = network.edges.get(key).unwrap();
            eb.iter().zip(ea).for_each(|(b, a)| {
                if *b >= 5 {
                    assert_eq!(b + 2, *a);
                } else {
                    assert_eq!(b, a);
                }
            });
        }
    }

    #[test]
    fn add_edge_between() {
        let mut network: Network<10> = Network::new(10, 10);
        network.edges.remove(&network.nodes[0].id);
        network.add_edge_between(0, 1);
        let connections = network.edges.get(&network.nodes[0].id).unwrap();
        assert_eq!(connections, &vec![1]);
    }

    #[test]
    fn remove_all_edges_to() {
        let mut network: Network<10> = Network::new(10, 10);
        network.add_nodes(NodeKind::Normal, 10);
        network.edges.remove(&network.nodes[0].id);
        network.edges.remove(&network.nodes[10].id);
        network.edges.remove(&network.nodes[20].id);
        network.add_edge_between(0, 20);
        network.add_edge_between(10, 20);
        // network.remove_all_edges_to(20);
        assert_eq!(network.edges_to_count.get(&network.nodes[20].id), None);
        assert_eq!(network.edges.get(&network.nodes[0].id).unwrap(), &vec![]);
        assert_eq!(network.edges.get(&network.nodes[10].id).unwrap(), &vec![]);
    }
}
