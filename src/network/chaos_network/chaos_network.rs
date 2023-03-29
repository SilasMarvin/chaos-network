use rand::distributions::{Uniform, WeightedIndex};
use rand::prelude::*;
use rand::Rng;

use serde::Serialize;

use std::boxed::Box;
use std::fs::DirBuilder;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::gradients::Gradients;
use crate::gradients::Tape;
use crate::network::{AdamOptimizer, Optimizer};
use crate::tensor_operations::{
    Tensor0DAdd, Tensor0DMul, Tensor1DAdd, Tensor1DMish, Tensor1DSplitOnAdd,
};
use crate::tensors::{Tensor0D, Tensor1D, WithTape, WithoutTape};

pub static NODE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NodeKind {
    Input,
    Normal,
    Leaf,
}

#[derive(Clone, Copy)]
pub enum ShiftDirection {
    Forward,
    Backward,
}

#[derive(Default)]
pub struct ChaosNetwork<const I: usize, const O: usize, const N: usize> {
    pub inputs_count: usize,
    pub leaves_count: usize,
    pub nodes: Vec<Node<N>>,
    pub tape: Tape<N>,
    pub input_connectivity_chance: f64,
}

#[derive(Clone)]
pub struct Node<const N: usize> {
    pub id: usize,
    pub weights: Vec<Tensor0D<N, WithTape>>,
    pub edges: Vec<usize>,
    pub edges_to_count: usize,
    pub kind: NodeKind,
    pub optimizer: Box<dyn Optimizer>,
}

impl<const I: usize, const O: usize, const N: usize> ChaosNetwork<I, O, N> {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let mut network: ChaosNetwork<I, O, N> = ChaosNetwork::default();
        network.input_connectivity_chance = 0.00035;
        network.add_nodes(NodeKind::Input, inputs);
        network.add_nodes(NodeKind::Leaf, outputs);
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
                    self.insert_node(NodeKind::Normal, node_index);
                    self.add_random_edge(node_index);
                    self.add_random_edge_to(node_index);
                }
            }
            NodeKind::Input => {
                // if self.leaves_count == 0 {
                //     panic!("This should be called after leaves are added for now")
                // }
                // let mut rng = rand::thread_rng();
                // for _i in 0..count {
                //     self.insert_node(NodeKind::Input, 0);
                //     self.inputs_count += 1;
                //     for ii in 0..self.leaves_count {
                //         if rng.gen::<f64>() < self.input_connectivity_chance {
                //             self.add_edge(0, self.nodes.len() - 1 - ii);
                //         }
                //     }
                // }
                if self.nodes.len() > 0 {
                    panic!("Adding the input nodes should be the first step")
                }
                for _i in 0..count {
                    self.insert_node(NodeKind::Input, 0);
                    self.inputs_count += 1;
                }
            }
            NodeKind::Leaf => {
                if self.inputs_count == 0 {
                    panic!("Adding leaves should be done after adding input nodes")
                }
                for _i in 0..count {
                    self.insert_node(NodeKind::Leaf, self.nodes.len());
                    self.add_random_edge_to(self.nodes.len() - 1);
                    self.leaves_count += 1;
                }
            }
        }
    }

    fn shift_all_edges_after(&mut self, after: usize, count: usize, direction: ShiftDirection) {
        self.nodes
            .iter_mut()
            .for_each(|n| n.shift_all_edges_after(after, count, direction));
    }

    fn insert_node(&mut self, kind: NodeKind, index: usize) {
        let new_node = Node::new(kind, &mut self.tape);
        self.nodes.insert(index, new_node);
        self.shift_all_edges_after(index, 1, ShiftDirection::Forward);
    }

    // NOTE This is broken
    // fn remove_node_with_no_edges_to_it(&mut self, index: usize) {
    //     while self.nodes[index].edges.len() > 0 {
    //         self.remove_edge(index, self.nodes[index].edges[0]);
    //     }
    //     let mut node = self.nodes.remove(index);
    //     node.remove_all_weights(&mut self.tape);
    //     self.shift_all_edges_after(index, 1, ShiftDirection::Backward);
    // }

    fn add_edge(&mut self, node_index_from: usize, node_index_to: usize) {
        self.nodes[node_index_from].add_edge(node_index_to, &mut self.tape);
        self.nodes[node_index_to].increment_edges_to_count();
    }

    // NOTE: This is broken
    // fn remove_edge(&mut self, node_index_from: usize, node_index_to: usize) {
    //     self.nodes[node_index_from].remove_edge(node_index_to, &mut self.tape);
    //     self.nodes[node_index_to].decrement_edges_to_count();
    //     if self.nodes[node_index_to].kind == NodeKind::Normal
    //         && self.nodes[node_index_to].edges_to_count == 0
    //     {
    //         self.remove_node_with_no_edges_to_it(node_index_to);
    //     }
    // }

    fn add_random_edge(&mut self, node_index_from: usize) {
        let mut rng = rand::thread_rng();
        let edges = &self.nodes[node_index_from].edges;
        let mut lock_break = 0;
        let node_index_to = loop {
            let index =
                rng.gen_range((node_index_from + 1).max(self.inputs_count)..self.nodes.len());
            if !edges.contains(&index) {
                break index;
            }
            if lock_break > 1000 {
                return;
            }
            lock_break += 1;
        };
        self.add_edge(node_index_from, node_index_to);
    }

    fn add_random_edge_to(&mut self, node_index_to: usize) {
        let mut rng = rand::thread_rng();
        let mut lock_break = 0;
        let node_index_from = loop {
            let index = rng.gen_range(0..node_index_to);
            let edges = &self.nodes[index].edges;
            if !edges.contains(&node_index_to) {
                break index;
            }
            if lock_break > 1000 {
                return;
            }
            lock_break += 1;
        };
        self.add_edge(node_index_from, node_index_to);
    }

    pub fn add_random_edges_for_random_nodes(&mut self, count: usize) {
        let mut rng = &mut thread_rng();
        let distribution = Uniform::from(0..self.nodes.len() - self.leaves_count);
        for _i in 0..count {
            let node_index_from = distribution.sample(&mut rng);
            self.add_random_edge(node_index_from);
        }
    }

    // NOTE This is broken
    // pub fn remove_weighted_edges(&mut self, count: usize) {
    //     let mut rng = &mut thread_rng();
    //     let distribution = Uniform::from(0..self.nodes.len() - self.leaves_count);
    //     for _i in 0..count {
    //         let node_index_from = distribution.sample(&mut rng);
    //         if self.nodes[node_index_from].edges.is_empty() {
    //             continue;
    //         }
    //         let edge_index = rng.gen_range(0..self.nodes[node_index_from].edges.len());
    //         let node_index_to = self.nodes[node_index_from].edges[edge_index];
    //         self.remove_edge(node_index_from, node_index_to);
    //     }
    // }

    pub fn get_edge_count(&self) -> usize {
        self.nodes.iter().fold(0, |acc, n| acc + n.edges.len())
    }

    pub fn forward_batch(&mut self, input: &[Tensor1D<N>; I]) -> Box<[Tensor1D<N, WithTape>; O]> {
        let mut output: Box<[Tensor1D<N, WithTape>; O]> = Box::new(
            (0..O)
                .map(|_i| Tensor1D::new([0.; N]))
                .collect::<Vec<Tensor1D<N, WithTape>>>()
                .try_into()
                .unwrap(),
        );
        let mut running_values: Vec<Option<Tensor1D<N, WithTape>>> = Vec::new();
        running_values.resize(self.nodes.len(), None);
        let nodes_len = self.nodes.len();
        for (i, node) in self.nodes.iter_mut().enumerate() {
            match node.kind {
                NodeKind::Input => {
                    if node.edges.is_empty() {
                        continue;
                    }
                    for (ii, edge) in node.edges.iter().enumerate() {
                        let mut x =
                            node.weights[ii].mul_left_by_reference(&input[i], &mut self.tape);
                        let running_value = &mut running_values[*edge];
                        running_values[*edge] = match running_value {
                            Some(rv) => Some(x.add(rv, &mut self.tape)),
                            None => Some(x),
                        }
                    }
                }
                NodeKind::Normal => {
                    let running_value = &mut running_values[i];
                    let mut go_in = match running_value.as_mut() {
                        Some(mut rv) => node.weights[0].add(&mut rv, &mut self.tape),
                        None => {
                            panic!("We should not be at a normal node that does not have a running value");
                        }
                    };
                    let go_in = go_in.mish(&mut self.tape);
                    let mut go_in = match node.edges.len() > 1 {
                        true => go_in.split_on_add(node.edges.len(), &mut self.tape),
                        _ => vec![go_in],
                    };
                    for (ii, edge) in node.edges.iter().enumerate() {
                        let mut x = node.weights[ii + 1].mul(&mut go_in[ii], &mut self.tape);
                        let running_value = &mut running_values[*edge];
                        running_values[*edge] = match running_value {
                            Some(rv) => Some(x.add(rv, &mut self.tape)),
                            None => Some(x),
                        }
                    }
                }
                NodeKind::Leaf => {
                    let mut val = std::mem::replace(&mut running_values[i], None)
                        .unwrap_or(Tensor1D::new([0.; N]));
                    let mut val = node.weights[0].add(&mut val, &mut self.tape);
                    let val = val.mish(&mut self.tape);
                    output[nodes_len - i - 1] = val;
                }
            }
        }
        output
    }

    pub fn forward_batch_no_grad(&mut self, input: &[Tensor1D<N>; I]) -> Box<[Tensor1D<N>; O]> {
        let mut output: [Tensor1D<N>; O] = (0..O)
            .map(|_i| Tensor1D::new([0.; N]))
            .collect::<Vec<Tensor1D<N>>>()
            .try_into()
            .unwrap();
        let mut running_values: Vec<Option<Tensor1D<N, WithoutTape>>> = Vec::new();
        running_values.resize(self.nodes.len(), None);
        let nodes_len = self.nodes.len();
        for (i, node) in self.nodes.iter_mut().enumerate() {
            match node.kind {
                NodeKind::Input => {
                    if node.edges.is_empty() {
                        continue;
                    }
                    for (ii, edge) in node.edges.iter().enumerate() {
                        let mut x =
                            node.weights[ii].mul_explicit_no_grad(&input[i], &mut self.tape);
                        let running_value = &mut running_values[*edge];
                        running_values[*edge] = match running_value {
                            Some(rv) => Some(x.add(rv, &mut self.tape).to_without_tape()),
                            None => Some(x.to_without_tape()),
                        }
                    }
                }
                NodeKind::Normal => {
                    let running_value = &mut running_values[i];
                    let mut go_in = match running_value.as_mut() {
                        Some(mut rv) => node.weights[0].add(&mut rv, &mut self.tape),
                        None => panic!(
                            "We should not be at a normal node that does not have a running value"
                        ),
                    };
                    let go_in = go_in.mish(&mut self.tape);
                    for (ii, edge) in node.edges.iter().enumerate() {
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
                    let mut val = std::mem::replace(&mut running_values[i], None)
                        .unwrap_or(Tensor1D::new([0.; N]));
                    let mut val = node.weights[0].add(&mut val, &mut self.tape);
                    let val = val.mish(&mut self.tape);
                    output[nodes_len - i - 1] = val;
                }
            }
        }
        Box::new(output)
    }

    pub fn execute_and_apply_gradients(&mut self) {
        let mut grads = self.tape.execute();
        for n in self.nodes.iter_mut() {
            n.apply_gradients(&mut grads);
        }
    }

    pub fn write_to_new_dir(&self) -> std::io::Result<()> {
        let mut write_out: Vec<(NodeKind, Vec<usize>, Vec<f64>)> = Vec::new();
        for n in &self.nodes {
            write_out.push((
                n.kind,
                n.edges.clone(),
                n.weights.iter().map(|w| w.data).collect(),
            ));
        }
        let write_out = serde_json::to_string(&write_out)?;
        let path: String = format!(
            "networks/{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        DirBuilder::new().recursive(true).create(&path).unwrap();
        let path: String = format!("{}/{}.json", path, "1");
        let mut file = File::create(path)?;
        write!(file, "{}", write_out)?;
        Ok(())
    }
}

impl<const N: usize> Node<N> {
    fn new(kind: NodeKind, tape: &mut Tape<N>) -> Self {
        match kind {
            NodeKind::Normal | NodeKind::Leaf => {
                let mut node = Self {
                    id: NODE_COUNT.fetch_add(1, Ordering::SeqCst),
                    weights: Vec::new(),
                    edges: Vec::new(),
                    edges_to_count: 0,
                    kind,
                    optimizer: Box::new(AdamOptimizer::default()),
                };
                node.add_weight(tape);
                node
            }
            _ => Self {
                id: NODE_COUNT.fetch_add(1, Ordering::SeqCst),
                weights: Vec::new(),
                edges: Vec::new(),
                edges_to_count: 0,
                kind,
                optimizer: Box::new(AdamOptimizer::default()),
            },
        }
    }

    fn add_edge(&mut self, node_index_to: usize, tape: &mut Tape<N>) {
        self.edges.push(node_index_to);
        self.add_weight(tape);
    }

    fn remove_edge(&mut self, node_index_to: usize, tape: &mut Tape<N>) {
        let index = self
            .edges
            .iter()
            .position(|ei| *ei == node_index_to)
            .unwrap();
        self.edges.remove(index);
        match self.kind {
            NodeKind::Normal => self.remove_weight(index + 1, tape),
            _ => self.remove_weight(index, tape),
        }
    }

    fn increment_edges_to_count(&mut self) {
        self.edges_to_count += 1;
    }

    fn decrement_edges_to_count(&mut self) {
        self.edges_to_count -= 1;
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

    fn shift_all_edges_after(&mut self, after: usize, count: usize, direction: ShiftDirection) {
        self.edges.iter_mut().for_each(|e| {
            if *e >= after {
                match direction {
                    ShiftDirection::Forward => *e += count,
                    ShiftDirection::Backward => *e -= count,
                }
            }
        });
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

    fn to_string(&self) -> String {
        let mut string = String::new();
        let k = match &self.kind {
            NodeKind::Input => 'I',
            NodeKind::Normal => 'N',
            NodeKind::Leaf => 'L',
        };
        string.push(k);
        let mut weights_skip = 0;
        if self.kind == NodeKind::Normal {
            string.push_str(&format!("{}|", self.weights[0].data));
            weights_skip = 1;
        }
        self.edges
            .iter()
            .zip(self.weights.iter().skip(weights_skip))
            .for_each(|(e, w)| string.push_str(&format!("{}|{}", e, w.data)));
        string
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_edge() {
        let mut network: ChaosNetwork<0, 10, 0> = ChaosNetwork::default();
        network.insert_node(NodeKind::Input, 0);
        network.insert_node(NodeKind::Normal, 1);
        network.add_edge(0, 1);
        assert_eq!(network.nodes[0].edges[0], 1);
        assert_eq!(network.nodes[1].edges_to_count, 1);
    }

    #[test]
    fn test_add_random_edge() {
        let mut network: ChaosNetwork<0, 10, 0> = ChaosNetwork::default();
        network.insert_node(NodeKind::Input, 0);
        network.insert_node(NodeKind::Normal, 1);
        network.insert_node(NodeKind::Normal, 2);
        network.add_random_edge(0);
        assert!(
            network.nodes[0].edges.len() == 1
                && (network.nodes[0].edges[0] == 1 || network.nodes[0].edges[0] == 2)
        );
    }

    #[test]
    fn test_add_random_edge_to() {
        let mut network: ChaosNetwork<0, 10, 0> = ChaosNetwork::default();
        network.insert_node(NodeKind::Input, 0);
        network.insert_node(NodeKind::Input, 1);
        network.insert_node(NodeKind::Normal, 2);
        network.add_random_edge_to(2);
        assert!(network.nodes[0].edges.len() == 1 || network.nodes[1].edges.len() == 1);
        if network.nodes[0].edges.len() == 1 {
            assert_eq!(network.nodes[0].edges[0], 2);
        } else {
            assert_eq!(network.nodes[1].edges[0], 2);
        }
        assert_eq!(network.nodes[2].edges_to_count, 1);
    }

    // #[test]
    // fn remove_edge() {
    //     let mut network: ChaosNetwork<0, 10, 0> = ChaosNetwork::default();
    //     network.insert_node(NodeKind::Input, 0);
    //     network.insert_node(NodeKind::Leaf, 1);
    //     network.add_edge(0, 1);
    //     network.remove_edge(0, 1);
    //     assert_eq!(network.nodes[0].edges.len(), 0);
    //     assert_eq!(network.nodes[1].edges_to_count, 0);
    // }
    //
    // #[test]
    // fn remove_node_with_no_edges_to_it() {
    //     // Base
    //     let mut network: ChaosNetwork<0, 10, 0> = ChaosNetwork::default();
    //     network.insert_node(NodeKind::Input, 0);
    //     network.insert_node(NodeKind::Normal, 1);
    //     network.add_edge(0, 1);
    //     network.remove_edge(0, 1);
    //     assert_eq!(network.nodes[0].edges.len(), 0);
    //     assert_eq!(network.nodes.len(), 1);
    //     // Base with correct shifts
    //     let mut network: ChaosNetwork<0, 10, 0> = ChaosNetwork::default();
    //     network.insert_node(NodeKind::Input, 0);
    //     network.insert_node(NodeKind::Normal, 1);
    //     network.insert_node(NodeKind::Normal, 2);
    //     network.insert_node(NodeKind::Normal, 3);
    //     network.add_edge(0, 1);
    //     network.add_edge(0, 2);
    //     network.add_edge(0, 3);
    //     network.remove_edge(0, 1);
    //     assert_eq!(network.nodes[0].edges, vec![1, 2]);
    //     // Recursive remove
    //     let mut network: ChaosNetwork<0, 10, 0> = ChaosNetwork::default();
    //     network.insert_node(NodeKind::Input, 0);
    //     network.insert_node(NodeKind::Normal, 1);
    //     network.insert_node(NodeKind::Normal, 2);
    //     network.insert_node(NodeKind::Normal, 3);
    //     let id_before = network.nodes[3].id;
    //     network.add_edge(0, 1);
    //     network.add_edge(1, 2);
    //     network.add_edge(1, 3);
    //     network.add_edge(0, 3);
    //     network.remove_edge(0, 1);
    //     assert_eq!(network.nodes[0].edges[0], 1);
    //     assert_eq!(network.nodes.len(), 2);
    //     assert_eq!(network.nodes[1].id, id_before);
    //     // Recursive remove test 2
    //     let mut network: ChaosNetwork<0, 10, 0> = ChaosNetwork::default();
    //     network.insert_node(NodeKind::Input, 0);
    //     network.insert_node(NodeKind::Normal, 1);
    //     network.insert_node(NodeKind::Normal, 2);
    //     network.insert_node(NodeKind::Normal, 3);
    //     let id_before = network.nodes[2].id;
    //     network.add_edge(0, 1);
    //     network.add_edge(1, 2);
    //     network.add_edge(1, 3);
    //     network.add_edge(0, 2);
    //     network.remove_edge(0, 1);
    //     assert_eq!(network.nodes[0].edges[0], 1);
    //     assert_eq!(network.nodes.len(), 2);
    //     assert_eq!(network.nodes[1].id, id_before);
    // }

    // #[test]
    // fn test_add_nodes() {
    //     // Base network with no normal nodes
    //     let mut network: ChaosNetwork<0, 10, 0> = ChaosNetwork::default();
    //     network.input_connectivity_chance = 1.0;
    //     network.add_nodes(NodeKind::Leaf, 10);
    //     network.add_nodes(NodeKind::Input, 10);
    //     assert_eq!(network.get_edge_count(), 100);
    //     // ChaosNetwork with normal nodes
    //     let mut network: ChaosNetwork<0, 10, 0> = ChaosNetwork::default();
    //     network.input_connectivity_chance = 1.0;
    //     network.add_nodes(NodeKind::Leaf, 10);
    //     network.add_nodes(NodeKind::Input, 10);
    //     // Get the node ids for all connections in the network
    //     let ids_of_connected_nodes_before: Vec<Vec<usize>> = network.nodes[0..10]
    //         .iter()
    //         .map(|n| {
    //             n.edges
    //                 .iter()
    //                 .map(|u| network.nodes[*u].id)
    //                 .collect::<Vec<usize>>()
    //         })
    //         .collect();
    //     network.add_nodes(NodeKind::Normal, 10);
    //     // Get the node ids for all connections in the network after normal nodes inserts
    //     let ids_of_connected_nodes_after: Vec<Vec<usize>> = network.nodes[0..10]
    //         .iter()
    //         .map(|n| {
    //             n.edges
    //                 .iter()
    //                 .map(|u| network.nodes[*u].id)
    //                 .collect::<Vec<usize>>()
    //         })
    //         .collect();
    //     // Make sure there are the expected number of connections
    //     assert_eq!(network.get_edge_count(), 30);
    //     // Make sure all of the old ids are still present in the input nodes connections
    //     ids_of_connected_nodes_before

    //         .zip(ids_of_connected_nodes_after.into_iter())
    //         .for_each(|(b, a)| {
    //             b.into_iter()
    //                 .for_each(|node_id| assert!(a.contains(&node_id)));
    //         });
    // }

    #[test]
    fn test_new_network() {
        let network: ChaosNetwork<10, 10, 10> = ChaosNetwork::new(10, 10);
        assert_eq!(network.inputs_count, 10);
        assert_eq!(network.leaves_count, 10);
    }
}
