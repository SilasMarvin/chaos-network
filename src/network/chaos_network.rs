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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
                    let go_in = input.pop().unwrap();
                    // let go_in = &mut input.remove(0)
                    //     + &mut (&mut node.weights[0] * &mut Tensor0D::new_without_tape(1.));
                    let mut go_in = match connections.len() > 1 {
                        true => go_in.split_on_add(connections.len()),
                        _ => vec![go_in],
                    };
                    for (ii, connection) in connections.iter().enumerate() {
                        let mut x = &mut go_in.pop().unwrap() * &mut node.weights[ii];
                        let running_value = &mut running_values[*connection];
                        running_values[*connection] = running_value + &mut x;
                    }
                }
                NodeKind::Normal => {
                    let connections = self.connections_to.get(&node.id).unwrap();
                    let mut go_in =
                        std::mem::replace(&mut running_values[i], Tensor0D::new_without_tape(0.));
                    // let mut go_in = &mut running_values[i]
                    //     + &mut (&mut node.weights[0] * &mut Tensor0D::new_without_tape(1.));
                    let go_in = Tensor0D::mish(&mut go_in);
                    let mut go_in = match connections.len() > 1 {
                        true => go_in.split_on_add(connections.len()),
                        _ => vec![go_in],
                    };
                    for (ii, connection) in connections.iter().enumerate() {
                        let mut x = &mut go_in.pop().unwrap() * &mut node.weights[ii];
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
        let new = match kind {
            NodeKind::Leaf => Self {
                id: NODE_COUNT.fetch_add(1, Ordering::SeqCst),
                weights: Vec::new(),
                kind,
                leaf_id: LEAF_COUNT.fetch_add(1, Ordering::SeqCst),
            },
            _ => {
                let mut node = Self {
                    id: NODE_COUNT.fetch_add(1, Ordering::SeqCst),
                    weights: Vec::new(),
                    kind,
                    leaf_id: 0,
                };
                // node.add_weight();
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
            let update = (0.01 * w_gradients.data * scale).clamp(-0.2, 0.2);
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
        let mut network = Network::new(10, 10, 10);
        // for n in network.nodes.iter_mut() {
        //     for w in n.weights.iter_mut() {
        //         match &n.kind {
        //             NodeKind::Input => {
        //                 println!("{:?}", w.id);
        //                 w.data = 0.5;
        //             }
        //             _ => {
        //                 w.data = 0.3;
        //             }
        //         }
        //     }
        // }
        network.dump_nodes_and_connections();
        let input: Vec<Tensor0D> = (0..784)
            .into_iter()
            // .map(|x| Tensor0D::new_without_tape((x % 10) as f64 / 10. - 0.5))
            .map(|x| Tensor0D::new_without_tape(0.5))
            .collect();
        network.set_mode(NetworkMode::Training);
        let mut output = network.forward(input);
        println!();
        println!("Output: {:?}", output);
        println!();
        // let mut loss = output
        //     .into_iter()
        //     .fold(Tensor0D::new_without_tape(1.), |mut acc, mut x| {
        //         &mut x * &mut acc
        //     });
        let mut loss = Tensor0D::nll(output, 0);
        println!();
        println!("Loss: {}", loss.data);
        println!();
        let mut grads = loss.backward();
        println!();
        for n in network.nodes.iter() {
            for w in n.weights.iter() {
                let g = grads.remove(w.id);
                println!("{:?}: {} {} - ", n.kind, w.id, g.data);
            }
        }
    }
}
