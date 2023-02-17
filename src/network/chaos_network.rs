use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use crate::gradients::Gradients;
use crate::gradients::Tape;
use crate::tensors::{Tensor, Tensor0D};

pub static NODE_COUNT: AtomicI32 = AtomicI32::new(0);
pub static LEAF_COUNT: AtomicI32 = AtomicI32::new(0);

#[derive(Debug, Clone, Copy)]
pub enum NodeKind {
    Input,
    Normal,
    Leaf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkMode {
    Training,
    Inference,
}

impl Default for NetworkMode {
    fn default() -> Self {
        Self::Inference
    }
}

#[derive(Default, Clone)]
pub struct Network {
    pub inputs_count: i32,
    pub leaves_count: i32,
    pub nodes: Vec<Node>,
    connections_to: HashMap<i32, Vec<usize>>,
    mode: NetworkMode,
    tape: Rc<RefCell<Tape>>,
}

#[derive(Clone)]
pub struct Node {
    pub id: i32,
    weights: Vec<Tensor0D>,
    running_value: Tensor0D,
    running_hits: i32,
    connected_from: i32,
    pub kind: NodeKind,
    leaf_id: i32,
}

impl Network {
    pub fn new(inputs_count: usize, normals_count: usize, leaves_count: usize) -> Self {
        let mut new_network = Self {
            inputs_count: inputs_count as i32,
            leaves_count: leaves_count as i32,
            nodes: Vec::new(),
            connections_to: HashMap::new(),
            mode: NetworkMode::Inference,
            tape: Rc::new(RefCell::new(Tape::new())),
        };

        let mut nodes: Vec<Node> = (0..inputs_count)
            .map(|_i| Node::new(NodeKind::Input))
            .collect();
        for _i in 0..inputs_count {
            new_network.nodes.push(nodes.remove(0));
        }

        let mut nodes: Vec<Node> = (0..normals_count)
            .map(|_i| Node::new(NodeKind::Normal))
            .collect();
        for _i in 0..normals_count {
            new_network.nodes.push(nodes.remove(0));
        }

        let mut nodes: Vec<Node> = (0..leaves_count)
            .map(|_i| Node::new(NodeKind::Leaf))
            .collect();
        for _i in 0..leaves_count {
            new_network.nodes.push(nodes.remove(0));
        }

        (0..inputs_count).into_iter().for_each(|i| {
            (0..normals_count)
                .into_iter()
                .for_each(|ii| new_network.add_connection_between(i, inputs_count + ii))
        });
        (0..normals_count).into_iter().for_each(|i| {
            (0..leaves_count).into_iter().for_each(|ii| {
                new_network
                    .add_connection_between(inputs_count + i, inputs_count + normals_count + ii)
            })
        });
        new_network
    }

    // pub fn add_nodes(&mut self, kind: NodeKind, count: i32) {
    //     match kind {
    //         NodeKind::Normal => {
    //             let node_index = self.batch_insert_normal_nodes(count as usize);
    //             for i in 0..(count as usize) {
    //                 self.add_normal_node_first_connection(node_index + i);
    //             }
    //             for i in 0..(count as usize) {
    //                 for _ii in 0..350 {
    //                     self.add_node_connection(node_index + i);
    //                 }
    //             }
    //         }
    //         NodeKind::Input => {
    //             self.inputs_count += count;
    //             let mut nodes: Vec<Node> = (0..count).map(|_i| Node::new(kind)).collect();
    //             let ids: Vec<i32> = nodes.iter().map(|n| n.id).collect();
    //             for _i in 0..count {
    //                 self.nodes.insert(0, nodes.remove(0));
    //             }
    //             self.shift_all_connections_after(0, count as usize);
    //             if self.leaves_count == 0 {
    //                 return;
    //             }
    //             for id in ids {
    //                 let mut new_connections: Vec<usize> = Vec::new();
    //                 for i in 0..self.leaves_count {
    //                     let index = (self.nodes.len() as i32 - i) as usize;
    //                     self.nodes[index].add_weight();
    //                     new_connections.push(index);
    //                 }
    //                 self.connections_to.insert(id, new_connections);
    //             }
    //         }
    //         NodeKind::Leaf => {
    //             self.leaves_count += count;
    //             for _i in 0..count {
    //                 self.nodes.push(Node::new(kind));
    //             }
    //             for i in 0..(self.inputs_count as usize) {
    //                 for ii in 0..(self.leaves_count as usize) {
    //                     self.add_connection_between(i, self.nodes.len() - ii - 1);
    //                 }
    //             }
    //         } // NodeKind::Leaf => {
    //           //     self.leaves_count += count;
    //           //     for _i in 0..count {
    //           //         self.nodes.push(Node::new(kind));
    //           //     }
    //           //     let mut inserted_node_index =
    //           //         self.batch_insert_normal_nodes((count * self.inputs_count) as usize);
    //           //     for i in 0..(self.inputs_count as usize) {
    //           //         // let node_id = self.nodes[i].id;
    //           //         // let mut new_connections = Vec::new();
    //           //         for ii in 0..(self.leaves_count as usize) {
    //           //             // new_connections.push(self.nodes.len() - ii - 1);
    //           //             // self.nodes[i].add_weight();
    //           //             self.add_node_connection_between(
    //           //                 inserted_node_index,
    //           //                 i,
    //           //                 self.nodes.len() - ii - 1,
    //           //             );
    //           //             inserted_node_index += 1;
    //           //         }
    //           //         // if let Some(connections) = self.connections_to.get_mut(&node_id) {
    //           //         //     connections.append(&mut new_connections);
    //           //         // } else {
    //           //         //     self.connections_to.insert(node_id, new_connections);
    //           //         // }
    //           //     }
    //           // }
    //     }
    // }

    fn batch_insert_normal_nodes(&mut self, count: usize) -> usize {
        let node_index = if self.nodes.len() as i32 - self.inputs_count - self.leaves_count > 0 {
            let mut rng = rand::thread_rng();
            let node_index = rng.gen_range(
                0..(self.nodes.len() - self.inputs_count as usize - self.leaves_count as usize),
            );
            node_index + self.inputs_count as usize
        } else {
            self.inputs_count as usize
        };
        for i in 0..count {
            self.nodes
                .insert(node_index + i as usize, Node::new(NodeKind::Normal));
        }
        self.shift_all_connections_after(node_index, count as usize);
        node_index
    }

    pub fn shift_all_connections_after(&mut self, after: usize, count: usize) {
        for (_key, value) in self.connections_to.iter_mut() {
            value.iter_mut().for_each(|u| {
                if *u >= after {
                    *u += count;
                }
            });
        }
    }

    fn get_connections_to(&self, node_index: usize) -> i32 {
        self.connections_to
            .iter()
            .fold(0, |acc, (key, value)| match value.contains(&node_index) {
                true => acc + 1,
                false => acc,
            })
    }

    fn swap_connections(&mut self, old_index: usize, new_index: usize) {
        for (_key, value) in self.connections_to.iter_mut() {
            value.iter_mut().for_each(|u| {
                if *u == old_index {
                    *u = new_index;
                }
            });
        }
    }

    fn add_connection_between(&mut self, node_index: usize, mut node2_index: usize) {
        // if self.get_connections_to(node2_index) >= 50 {
        //     let new_node = Node::new(NodeKind::Normal);
        //     let new_node_id = new_node.id;
        //     let new_node_index = node2_index.min(self.nodes.len() - self.leaves_count as usize);
        //     self.nodes.insert(new_node_index, new_node);
        //     self.shift_all_connections_after(new_node_index, 1);
        //     self.swap_connections(node2_index + 1, new_node_index);
        //     self.connections_to
        //         .insert(new_node_id, vec![node2_index + 1]);
        //     self.nodes[new_node_index].add_weight();
        //     node2_index = new_node_index;
        // }
        match self.connections_to.get_mut(&self.nodes[node_index].id) {
            Some(connections) => connections.push(node2_index),
            None => {
                self.connections_to
                    .insert(self.nodes[node_index].id, vec![node2_index]);
            }
        };
        self.nodes[node_index].add_weight();
    }

    fn add_node_connection(&mut self, node_index: usize) {
        let mut rng = rand::thread_rng();
        // Unsigned should not go below 0, so this should be fine
        let node2_index = rng.gen_range(node_index + 1..self.nodes.len());
        self.add_connection_between(node_index, node2_index);
    }

    fn add_normal_node_first_connection(&mut self, node_index: usize) {
        let mut rng = rand::thread_rng();
        let node1_index = rng.gen_range(0..node_index);
        // Unsigned should not go below 0, so this should be fine
        let node2_index = rng.gen_range(node_index + 1..self.nodes.len());
        self.add_node_connection_between(node_index, node1_index, node2_index);
    }

    fn add_node_connection_between(
        &mut self,
        node_index: usize,
        node1_index: usize,
        node2_index: usize,
    ) {
        match self.connections_to.get_mut(&self.nodes[node1_index].id) {
            Some(connections) => connections.push(node_index),
            None => {
                self.connections_to
                    .insert(self.nodes[node1_index].id, vec![node_index]);
            }
        };
        self.nodes[node1_index].add_weight();
        match self.connections_to.get_mut(&self.nodes[node_index].id) {
            Some(connections) => connections.push(node2_index),
            None => {
                self.connections_to
                    .insert(self.nodes[node_index].id, vec![node2_index]);
            }
        };
        self.nodes[node_index].add_weight();
    }

    pub fn forward(&mut self, mut input: Vec<Tensor0D>) -> Vec<Tensor0D> {
        let mut output: Vec<Tensor0D> = Vec::with_capacity(self.leaves_count as usize);
        output.resize(self.leaves_count as usize, Tensor0D::new_without_tape(0.));
        let mut running_values: Vec<Tensor0D> = Vec::with_capacity(self.nodes.len());
        running_values.resize(self.nodes.len(), Tensor0D::new_without_tape(0.));
        for (i, node) in self.nodes.iter_mut().enumerate() {
            match node.kind {
                NodeKind::Input => {
                    let connections = self.connections_to.get(&node.id).unwrap();
                    let go_in = &mut input.remove(0)
                        + &mut (&mut node.weights[0] * &mut Tensor0D::new_without_tape(1.));
                    let mut go_in = match connections.len() > 1 {
                        true => go_in.split_on_add(connections.len()),
                        _ => vec![go_in],
                    };
                    for (ii, connection) in connections.iter().enumerate() {
                        let mut x = &mut go_in.remove(0) * &mut node.weights[ii + 1];
                        let running_value = &mut running_values[*connection];
                        running_values[*connection] = running_value + &mut x;
                    }
                }
                NodeKind::Normal => {
                    let connections = self.connections_to.get(&node.id).unwrap();
                    let mut go_in = &mut running_values[i]
                        + &mut (&mut node.weights[0] * &mut Tensor0D::new_without_tape(1.));
                    let go_in = Tensor0D::mish(&mut go_in);
                    let mut go_in = match connections.len() > 1 {
                        true => go_in.split_on_add(connections.len()),
                        _ => vec![go_in],
                    };
                    for (ii, connection) in connections.iter().enumerate() {
                        let mut x = &mut go_in.remove(0) * &mut node.weights[ii + 1];
                        let running_value = &mut running_values[*connection];
                        running_values[*connection] = running_value + &mut x;
                    }
                }
                NodeKind::Leaf => {
                    output[node.leaf_id as usize] =
                        std::mem::replace(&mut running_values[i], Tensor0D::new_without_tape(0.));
                }
            }
        }
        output
    }

    pub fn morph(&mut self) {
        let mut rng = rand::thread_rng();
        let remove_count = rng.gen_range(0..(self.get_connection_count() as f32 * 0.03) as usize);
        self.remove_connections(remove_count);
        let add_count = rng.gen_range(0..(self.get_connection_count() as f32 * 0.05) as usize);
        self.add_random_connections(add_count);
    }

    pub fn add_random_connections(&mut self, count: usize) {
        let mut rng = rand::thread_rng();
        for _i in 0..count {
            let node_index = rng.gen_range(0..self.nodes.len() - self.leaves_count as usize);
            self.add_node_connection(node_index);
        }
    }

    pub fn remove_connections(&mut self, count: usize) {
        let mut rng = rand::thread_rng();
        let weights = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_i, n)| n.weights.len() > 2)
            .map(|(i, n)| {
                n.weights
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(move |(ii, w)| (i, ii, w.data))
            })
            .flatten()
            .collect::<Vec<(usize, usize, f64)>>();
        if weights.len() == 0 {
            return;
        }
        let weighted = WeightedIndex::new(weights.iter().map(|w| w.2.powi(2) + 1.0)).unwrap();
        let mut sampled = HashSet::new();
        for _i in 0..count {
            let item = weights[weighted.sample(&mut rng)];
            if sampled.contains(&item.0) {
                continue;
            }
            sampled.insert(item.0);
            self.connections_to
                .get_mut(&self.nodes[item.0].id)
                .unwrap()
                .remove(item.1 - 1);
            self.nodes[item.0].weights.remove(item.1);
        }
    }

    // pub fn backward(&mut self, mut loss: Tensor0D) {
    //     let mut gradients = loss.backward();
    //     for n in self.nodes.iter_mut() {
    //         n.apply_gradients(&mut gradients);
    //     }
    // }

    pub fn apply_gradients(&mut self, mut gradients: Gradients, scale: f64) {
        for n in self.nodes.iter_mut() {
            n.apply_gradients(&mut gradients, scale);
        }
    }

    pub fn set_mode(&mut self, mode: NetworkMode) {
        self.mode = mode;
        for n in self.nodes.iter_mut() {
            if mode == NetworkMode::Training {
                n.set_mode(mode, Some(self.tape.clone()));
            } else {
                n.set_mode(mode, None);
            }
        }
    }

    pub fn get_connection_count(&self) -> i32 {
        self.connections_to
            .iter()
            .fold(0, |acc, (_key, value)| acc + value.len() as i32)
    }

    pub fn dump_nodes_and_connections(&self) {
        println!("Nodes");
        for (i, n) in self.nodes.iter().enumerate() {
            println!("{} {:?}", i, n);
        }
        println!("Connections");
        for (key, val) in self.connections_to.iter() {
            println!("Id: {} - Connected To: {:?}", key, val);
        }
    }
}

impl Node {
    pub fn new(kind: NodeKind) -> Self {
        let mut new = match kind {
            NodeKind::Leaf => Self {
                id: NODE_COUNT.fetch_add(1, Ordering::SeqCst),
                weights: Vec::new(),
                running_value: Tensor0D::new_without_tape(0.),
                running_hits: 0,
                kind,
                connected_from: 0,
                leaf_id: LEAF_COUNT.fetch_add(1, Ordering::SeqCst),
            },
            _ => {
                let mut node = Self {
                    id: NODE_COUNT.fetch_add(1, Ordering::SeqCst),
                    weights: Vec::new(),
                    running_value: Tensor0D::new_without_tape(0.),
                    running_hits: 0,
                    kind,
                    connected_from: 0,
                    leaf_id: 0,
                };
                node.add_weight();
                node
            }
        };
        new
    }

    pub fn add_weight(&mut self) {
        let mut rng = rand::thread_rng();
        let w = (rng.gen::<f64>() - 0.5) / 10.;
        self.weights.push(Tensor0D::new_without_tape(w));
    }

    fn apply_gradients(&mut self, gradients: &mut Gradients, scale: f64) {
        for w in self.weights.iter_mut() {
            let w_gradients = gradients.remove(w.id);
            // Do a little gradient clipping
            let update = (0.0025 * w_gradients.data * scale).clamp(-0.2, 0.2);
            w.data -= update;
            // w.reset_tape();
        }
    }

    fn set_mode(&mut self, mode: NetworkMode, tape: Option<Rc<RefCell<Tape>>>) {
        match mode {
            NetworkMode::Training => {
                for w in self.weights.iter_mut() {
                    w.set_tape(tape.clone());
                }
            }
            NetworkMode::Inference => {
                for w in self.weights.iter_mut() {
                    w.clear_tape();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        let mut network = Network::new(784, 512, 10);
        let input: Vec<Tensor0D> = (0..784)
            .into_iter()
            .map(|x| Tensor0D::new_without_tape(1.))
            .collect();
        let mut output = network.forward(input);
        println!("Output: {:?}", output);
    }

    // #[test]
    // fn test_forward_without_splits() {
    //     let mut network = Network::new();
    //     network.add_nodes(NodeKind::Input, 2);
    //     network.add_nodes(NodeKind::Leaf, 2);
    //     network.dump_nodes_and_connections();
    //     let mut weights: Vec<f64> = vec![0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.23, -0.23];
    //     network.nodes.iter_mut().for_each(|n| {
    //         n.weights.iter_mut().for_each(|w| {
    //             w.data = weights.remove(0);
    //         });
    //     });
    //     network.set_mode(NetworkMode::Training);
    //     let input = vec![
    //         Tensor0D::new_without_tape(0.1),
    //         Tensor0D::new_without_tape(-0.1),
    //     ];
    //     let mut output = network.forward(input);
    //     // Derived from pytorch
    //     // assert_eq!(output[0].data, -0.001894211);
    //     // assert_eq!(output[1].data, -0.001825735);
    //     println!("output: {:?}", output);
    //
    //     // let mut loss = Tensor0D::nll(output, 0);
    //     // assert_eq!(loss.data, 0.6944052);
    //     // let mut gradients = loss.backward();
    //     // let mut expected_grads = vec![
    //     //     0.005211302079260349,
    //     //     -0.007229849696159363,
    //     //     -0.002944408217445016,
    //     //     -0.0030677486211061478,
    //     //     0.0030197836458683014,
    //     //     -0.0004361033788882196,
    //     //     0.0032564920838922262,
    //     //     -0.0001915280445246026,
    //     //     -0.005503247492015362,
    //     //     -0.006071555893868208,
    //     // ];
    //
    //     let mut gradients = output[0].backward();
    //
    //     network.nodes.iter().for_each(|n| {
    //         n.weights.iter().for_each(|w| {
    //             let grad = gradients.remove(w.id);
    //             println!("grads: {}", grad.data);
    //             // assert_eq!(grad.data, expected_grads.remove(0))
    //         });
    //     });
    // }

    // #[test]
    // fn test_forward() {
    //     let mut network = Network::new();
    //     network.add_nodes(NodeKind::Input, 2);
    //     network.add_nodes(NodeKind::Leaf, 2);
    //     let new_node = Node::new(NodeKind::Normal);
    //     network.nodes.insert(2, new_node);
    //     network.shift_all_connections_after(2, 1);
    //     let new_node = Node::new(NodeKind::Normal);
    //     network.nodes.insert(3, new_node);
    //     network.shift_all_connections_after(3, 1);
    //     network.add_node_connection_between(2, 0, 5);
    //     network.add_node_connection_between(3, 0, 5);
    //     network.add_connection_between(2, 4);
    //     network.add_connection_between(2, 3);
    //     network.add_connection_between(3, 4);
    //     network.dump_nodes_and_connections();
    //     let mut weights: Vec<f64> = vec![
    //         0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.23, -0.23, 0.1, -0.1, 0.13, -0.13, 0.17, -0.17,
    //         0.14,
    //     ];
    //     network.nodes.iter_mut().for_each(|n| {
    //         n.weights.iter_mut().for_each(|w| {
    //             w.data = weights.remove(0);
    //         });
    //     });
    //     network.set_mode(NetworkMode::Training);
    //     let input = vec![
    //         Tensor0D::new_without_tape(0.1),
    //         Tensor0D::new_without_tape(-0.1),
    //     ];
    //     let mut output = network.forward(input);
    //     // Derived from pytorch
    //     // assert_eq!(output[0].data, -0.0027088919);
    //     // assert_eq!(output[1].data, -0.00019446987);
    //     println!("output: {:?}", output);
    //
    //     let mut loss = Tensor0D::nll(output, 0);
    //     // assert_eq!(loss.data, 0.6944052);
    //     println!("loss: {:?}", loss);
    //     let mut gradients = loss.backward();
    //     let mut expected_grads = vec![
    //         0.005211302079260349,
    //         -0.007229849696159363,
    //         -0.002944408217445016,
    //         -0.0030677486211061478,
    //         0.0030197836458683014,
    //         -0.0004361033788882196,
    //         0.0032564920838922262,
    //         -0.0001915280445246026,
    //         -0.005503247492015362,
    //         -0.006071555893868208,
    //         -0.006071555893868208,
    //         -0.006071555893868208,
    //         -0.006071555893868208,
    //         -0.006071555893868208,
    //         -0.006071555893868208,
    //     ];
    //
    //     network.nodes.iter().for_each(|n| {
    //         n.weights.iter().for_each(|w| {
    //             let grad = gradients.remove(w.id);
    //             println!(
    //                 "grads: {:.32} - expected: {}",
    //                 grad.data,
    //                 expected_grads.remove(0)
    //             );
    //             // assert_eq!(grad.data, expected_grads.remove(0))
    //         });
    //     });
    // }

    // #[test]
    // fn test_forward_rigid() {
    //     let mut w1 = Tensor0D::new_with_tape(0.1);
    //     let mut w2 = Tensor0D::new_with_tape(-0.1);
    //     let mut w3 = Tensor0D::new_with_tape(0.2);
    //     let mut w4 = Tensor0D::new_with_tape(-0.2);
    //     let mut w5 = Tensor0D::new_with_tape(0.15);
    //     let mut w6 = Tensor0D::new_with_tape(-0.15);
    //     let mut w7 = Tensor0D::new_with_tape(0.23);
    //
    //     let mut in1 = Tensor0D::split_on_add(Tensor0D::new_without_tape(0.1), 3);
    //     let mut in2 = Tensor0D::split_on_add(Tensor0D::new_without_tape(-0.1), 2);
    //
    //     let mut n1 = &mut in1[0] * &mut w5;
    //     let mut n1 = Tensor0D::split_on_add(n1, 2);
    //
    //     let mut o1 = &mut in1[1] * &mut w1;
    //     let mut o1 = &mut o1 + &mut (&mut in2[0] * &mut w3);
    //     let mut o1 = &mut o1 + &mut (&mut n1[0] * &mut w6);
    //
    //     let mut o2 = &mut in1[2] * &mut w2;
    //     let mut o2 = &mut o2 + &mut (&mut in2[1] * &mut w4);
    //     let mut o2 = &mut o2 + &mut (&mut n1[1] * &mut w7);
    //
    //     println!();
    //     println!("Output: {} - {}", o1.data, o2.data);
    //     let mut loss = Tensor0D::nll(vec![o1, o2], 0);
    //     println!();
    //     println!("Loss dump: {:?}", loss);
    //     println!("Loss: {:.32}", loss.data);
    //
    //     let mut gradients = loss.backward();
    //     println!("w1: {:.32}", gradients.remove(w1.id).data);
    //     println!("w2: {:.32}", gradients.remove(w2.id).data);
    //     println!("w3: {:.32}", gradients.remove(w3.id).data);
    //     println!("w4: {:.32}", gradients.remove(w4.id).data);
    //     println!("w5: {:.32}", gradients.remove(w5.id).data);
    //     println!("w6: {:.32}", gradients.remove(w6.id).data);
    //     println!("w7: {:.32}", gradients.remove(w7.id).data);
    // }

    // #[test]
    // fn test_forward_rigid() {
    //     let mut w1 = Tensor0D::new_with_tape(0.1);
    //     let mut w2 = Tensor0D::new_with_tape(-0.1);
    //     let mut w3 = Tensor0D::new_with_tape(0.2);
    //     let mut w4 = Tensor0D::new_with_tape(-0.2);
    //     let mut w5 = Tensor0D::new_with_tape(0.15);
    //     let mut w6 = Tensor0D::new_with_tape(-0.15);
    //     let mut w7 = Tensor0D::new_with_tape(0.23);
    //     let mut w8 = Tensor0D::new_with_tape(-0.23);
    //     let mut w9 = Tensor0D::new_with_tape(0.1);
    //     let mut w10 = Tensor0D::new_with_tape(-0.1);
    //
    //     let mut in1 = Tensor0D::split_on_add(Tensor0D::new_without_tape(0.1), 2);
    //     let mut in2 = Tensor0D::split_on_add(Tensor0D::new_without_tape(-0.1), 2);
    //
    //     let n1 = Tensor0D::mish(&mut (&mut in1[0] * &mut w1));
    //     let mut n1 = Tensor0D::split_on_add(n1, 2);
    //     let n2 = Tensor0D::mish(&mut (&mut (&mut in1[1] * &mut w2) + &mut (&mut n1[0] * &mut w6)));
    //     let mut n2 = Tensor0D::split_on_add(n2, 2);
    //     let mut n3 =
    //         Tensor0D::mish(&mut (&mut (&mut in2[0] * &mut w3) + &mut (&mut n2[0] * &mut w8)));
    //     let mut n4 = Tensor0D::mish(&mut (&mut in2[1] * &mut w4));
    //
    //     let mut o1 = &mut (&mut n2[1] * &mut w7) + &mut (&mut n4 * &mut w10);
    //     let mut o2 = &mut (&mut n1[1] * &mut w5) + &mut (&mut n3 * &mut w9);
    //
    //     let mut loss = Tensor0D::nll(vec![o1, o2], 0);
    //     println!("Loss dump: {:?}", loss);
    //     println!("Loss: {:.32}", loss.data);
    //
    //     let mut gradients = loss.backward();
    //     println!("w1: {:.32}", gradients.remove(w1.id).data);
    //     println!("w2: {:.32}", gradients.remove(w2.id).data);
    //     println!("w3: {:.32}", gradients.remove(w3.id).data);
    //     println!("w4: {:.32}", gradients.remove(w4.id).data);
    //     println!("w5: {:.32}", gradients.remove(w5.id).data);
    //     println!("w6: {:.32}", gradients.remove(w6.id).data);
    //     println!("w7: {:.32}", gradients.remove(w7.id).data);
    //     println!("w8: {:.32}", gradients.remove(w8.id).data);
    //     println!("w9: {:.32}", gradients.remove(w9.id).data);
    //     println!("w10: {:.32}", gradients.remove(w10.id).data);
    // }
}
