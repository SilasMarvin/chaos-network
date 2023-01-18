use rand::prelude::*;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use crate::gradients::Gradients;
use crate::tensors::{Tensor, Tensor0D};

pub static NODE_COUNT: AtomicI32 = AtomicI32::new(0);
pub static LEAF_COUNT: AtomicI32 = AtomicI32::new(0);

#[derive(Debug, Clone, Copy)]
pub enum NodeKind {
    Input,
    Normal,
    Leaf,
}

#[derive(Debug, Clone, Copy)]
pub enum NetworkMode {
    Training,
    Inference,
}

pub struct Network {
    inputs: Vec<Rc<RefCell<Node>>>,
    pub nodes: Vec<Rc<RefCell<Node>>>,
    pub leaves: Vec<Rc<RefCell<Node>>>,
    mode: NetworkMode,
}

pub struct Node {
    id: i32,
    pub weights: Vec<Tensor0D>,
    connections: Vec<Rc<RefCell<Node>>>,
    running_value: Tensor0D,
    running_hits: i32,
    connected_from: i32,
    kind: NodeKind,
    leaf_id: i32,
}

impl Network {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            nodes: Vec::new(),
            leaves: Vec::new(),
            mode: NetworkMode::Inference,
        }
    }

    pub fn add_node(&mut self, kind: NodeKind) {
        let node = Node::new(kind);
        let node = Rc::new(RefCell::new(node));
        self.nodes.push(node.clone());
        match kind {
            NodeKind::Normal => {
                self.add_normal_node_first_connection(node.clone());
            }
            NodeKind::Input => {
                self.inputs.push(node.clone());
                for n in &self.leaves {
                    (*node).borrow_mut().add_connection(n.clone());
                }
            }
            NodeKind::Leaf => {
                self.leaves.push(node.clone());
                for n in &self.inputs {
                    (**n).borrow_mut().add_connection(node.clone());
                }
            }
        }
        for _i in 0..5 {
            self.connect_node(node.clone());
        }
    }

    pub fn remove_node(&mut self, node: Rc<RefCell<Node>>) {
        let index = self.nodes.iter().position(|n| Rc::ptr_eq(n, &node));
        if let Some(i) = index {
            self.nodes.remove(i);
        }
    }

    pub fn add_random_connections(&mut self, connection_count: usize) {
        let mut rng = rand::thread_rng();
        let distribution = Uniform::from(0..self.nodes.len());
        let mut found = 0;
        loop {
            let choice = distribution.sample(&mut rng);
            let node = self.nodes[choice].clone();
            let kind = node.borrow().kind;
            match kind {
                NodeKind::Leaf => continue,
                _ => self.connect_node(node.clone()),
            };
            found += 1;
            if found == connection_count {
                break;
            }
        }
    }

    fn connect_node(&self, node: Rc<RefCell<Node>>) {
        let mut rng = rand::thread_rng();
        let mut order: Vec<usize> = (0..self.nodes.len()).collect();
        order.shuffle(&mut rng);
        for i in order {
            if Self::try_connect_node(node.clone(), self.nodes[i].clone()) {
                break;
            }
        }
    }

    fn try_connect_node(node: Rc<RefCell<Node>>, connect_node: Rc<RefCell<Node>>) -> bool {
        let kind = node.borrow().kind;
        match kind {
            NodeKind::Input => {
                if Rc::ptr_eq(&connect_node, &node)
                    || matches!(connect_node.borrow().kind, NodeKind::Input)
                {
                    false
                } else {
                    (*node).borrow_mut().add_connection(connect_node);
                    true
                }
            }
            NodeKind::Leaf => {
                if Rc::ptr_eq(&connect_node, &node)
                    || matches!(connect_node.borrow().kind, NodeKind::Leaf)
                {
                    false
                } else {
                    (*connect_node).borrow_mut().add_connection(node);
                    true
                }
            }
            NodeKind::Normal => {
                if Rc::ptr_eq(&node, &connect_node)
                    || matches!(connect_node.borrow().kind, NodeKind::Input)
                    || connect_node
                        .borrow()
                        .is_node_downstream(&node, &mut HashMap::new())
                {
                    false
                } else {
                    (*node).borrow_mut().add_connection(connect_node);
                    true
                }
            }
        }
    }

    fn add_normal_node_first_connection(&mut self, node: Rc<RefCell<Node>>) {
        let mut rng = rand::thread_rng();
        let mut order1: Vec<usize> = (0..self.nodes.len()).collect();
        let mut order2: Vec<usize> = (0..self.nodes.len()).collect();
        order1.shuffle(&mut rng);
        order2.shuffle(&mut rng);
        for i in order1 {
            for ii in &order2 {
                if Self::try_add_normal_node_first_connection(
                    node.clone(),
                    self.nodes[i].clone(),
                    self.nodes[*ii].clone(),
                ) {
                    return;
                }
            }
        }
    }

    fn try_add_normal_node_first_connection(
        node: Rc<RefCell<Node>>,
        connect_node1: Rc<RefCell<Node>>,
        connect_node2: Rc<RefCell<Node>>,
    ) -> bool {
        if Rc::ptr_eq(&connect_node1, &connect_node2)
            || Rc::ptr_eq(&connect_node1, &node)
            || Rc::ptr_eq(&connect_node2, &node)
            || matches!(connect_node1.borrow().kind, NodeKind::Leaf)
            || matches!(connect_node2.borrow().kind, NodeKind::Input)
            || connect_node2
                .borrow()
                .is_node_downstream(&connect_node1, &mut HashMap::new())
        {
            false
        } else {
            (*connect_node1).borrow_mut().add_connection(node.clone());
            (*node).borrow_mut().add_connection(connect_node2);
            true
        }
    }

    pub fn forward(&mut self, mut input: Vec<Tensor0D>) -> Vec<Tensor0D> {
        let mut output: Vec<Tensor0D> = Vec::with_capacity(self.leaves.len());
        output.resize(self.leaves.len(), Tensor0D::new_without_tape(0.));
        for (_i, n) in self.inputs.iter().enumerate() {
            (**n).borrow_mut().forward(input.remove(0), &mut output);
        }
        output
    }

    pub fn backward(&mut self, mut loss: Tensor0D) {
        let mut gradients = loss.backward();
        let mut visited = HashMap::new();
        for n in self.inputs.iter() {
            (**n)
                .borrow_mut()
                .apply_gradients(&mut gradients, &mut visited);
        }
    }

    pub fn morph(&mut self, morph_percent: f32) {
        let connections = self.nodes.iter().fold(Vec::new(), |mut acc, n| {
            for i in 0..n.borrow().connections.len() {
                acc.push((
                    n.clone(),
                    n.borrow().connections[i].clone(),
                    n.borrow().weights[i].data,
                ));
            }
            acc
        });
        let mut rng = thread_rng();
        let choice_count = self.nodes.len() as f32 * (morph_percent / 100.);
        let choices = connections
            .choose_multiple_weighted(&mut rng, choice_count as usize, |c| c.2.abs())
            .unwrap();
        for c in choices {
            if c.0.borrow().connections.len() == 1 {
                continue;
            } else if c.1.borrow().connected_from == 1 {
                for n in &c.1.borrow().connections {
                    if !c.0.borrow().is_connected_to(n.clone()) {
                        (*c.0).borrow_mut().add_connection(n.clone());
                    }
                }
                self.remove_node(c.1.clone());
            } else {
                (*c.0).borrow_mut().remove_connection(c.1.clone());
            }
        }
        self.add_random_connections(choice_count as usize);
    }

    pub fn grow(&mut self, growth_percent: f32) {
        let grow_count = self.nodes.len() as f32 * ((growth_percent / 2.) / 100.);
        println!("Adding nodes and connections: {}", grow_count as usize);
        for _i in 0..grow_count as usize {
            self.add_node(NodeKind::Normal);
        }
        self.add_random_connections(grow_count as usize);
    }

    pub fn set_mode(&mut self, mode: NetworkMode) {
        // if matches!(self.mode, mode) {
        //     return;
        // }
        self.mode = mode;
        let mut visited = HashMap::new();
        for n in &self.inputs {
            (**n).borrow_mut().set_mode(mode, &mut visited);
        }
    }

    pub fn get_connection_count(&self) -> i32 {
        self.nodes
            .iter()
            .fold(0, |acc, n| acc + n.borrow().connections.len() as i32)
    }

    pub fn dump_nodes_and_connections(&self) {
        for (i, n) in self.nodes.iter().enumerate() {
            println!("{} {:?}", i, n);
        }
    }
}

impl Node {
    pub fn new(kind: NodeKind) -> Self {
        let mut new = match kind {
            NodeKind::Leaf => Self {
                id: NODE_COUNT.fetch_add(1, Ordering::SeqCst),
                weights: Vec::new(),
                connections: Vec::new(),
                running_value: Tensor0D::new_without_tape(0.),
                running_hits: 0,
                kind,
                connected_from: 0,
                leaf_id: LEAF_COUNT.fetch_add(1, Ordering::SeqCst),
            },
            _ => Self {
                id: NODE_COUNT.fetch_add(1, Ordering::SeqCst),
                weights: Vec::new(),
                connections: Vec::new(),
                running_value: Tensor0D::new_without_tape(0.),
                running_hits: 0,
                kind,
                connected_from: 0,
                leaf_id: 0,
            },
        };
        let mut rng = rand::thread_rng();
        let w = rng.gen::<f32>() / 10.;
        new.weights.push(Tensor0D::new_without_tape(w));
        new
    }

    fn forward(&mut self, mut input: Tensor0D, output: &mut Vec<Tensor0D>) {
        self.running_hits += 1;
        let mut x = &mut self.running_value + &mut input;
        // If we do not have all of our data yet, comeback later
        if self.running_hits < self.connected_from {
            self.running_value = x;
            return;
        }
        // If we are done, reset the middle values
        self.running_value = Tensor0D::new_without_tape(0.);
        self.running_hits = 0;
        // If we are a leaf, push the value, else call forward on our connections
        match &self.kind {
            NodeKind::Leaf => {
                output[self.leaf_id as usize] = x;
            }
            _ => {
                x = &mut x
                    + &mut (&mut Tensor0D::new_without_tape(1.)
                        * self.weights.first_mut().unwrap());
                if matches!(self.kind, NodeKind::Normal) {
                    x = Tensor0D::mish(&mut x);
                }
                for (i, n) in self.connections.iter().enumerate() {
                    (**n)
                        .borrow_mut()
                        .forward(&mut self.weights[i + 1] * &mut x, output);
                }
            }
        };
    }

    fn apply_gradients(&mut self, gradients: &mut Gradients, visited: &mut HashMap<i32, bool>) {
        if *visited.get(&self.id).unwrap_or(&false) {
            return;
        }
        visited.insert(self.id, true);
        for w in self.weights.iter_mut() {
            let w_gradients = gradients.remove::<Tensor0D>(w.id);
            w.data -= 0.0025 * w_gradients.data;
            w.reset_tape();
        }
        for n in &self.connections {
            (**n).borrow_mut().apply_gradients(gradients, visited);
        }
    }

    fn is_node_downstream(
        &self,
        downstream_node: &Rc<RefCell<Node>>,
        visited: &mut HashMap<i32, bool>,
    ) -> bool {
        if self.id == downstream_node.borrow().id {
            true
        } else if *visited.get(&self.id).unwrap_or(&false) || self.connections.is_empty() {
            false
        } else {
            visited.insert(self.id, true);
            self.connections.iter().fold(false, |acc, node| {
                if acc {
                    acc
                } else {
                    node.borrow().is_node_downstream(downstream_node, visited)
                }
            })
        }
    }

    fn is_connected_to(&self, node: Rc<RefCell<Node>>) -> bool {
        self.connections
            .iter()
            .any(|c| c.borrow().id == node.borrow().id)
    }

    fn add_connection(&mut self, new_node: Rc<RefCell<Node>>) {
        let mut rng = rand::thread_rng();
        let w = rng.gen::<f32>() / 10.;
        (*new_node).borrow_mut().connected_from += 1;
        self.connections.push(new_node);
        self.weights.push(Tensor0D::new_without_tape(w));
    }

    fn remove_connection(&mut self, node: Rc<RefCell<Node>>) {
        (*node).borrow_mut().connected_from -= 1;
        let connection_index = self
            .connections
            .iter()
            .position(|c| c.borrow().id == node.borrow().id)
            .unwrap();
        self.weights.remove(connection_index);
        self.connections.remove(connection_index);
    }

    fn set_mode(&mut self, mode: NetworkMode, visited: &mut HashMap<i32, bool>) {
        if *visited.get(&self.id).unwrap_or(&false) {
            return;
        }
        visited.insert(self.id, true);
        match mode {
            NetworkMode::Training => {
                for w in &mut self.weights {
                    w.reset_tape();
                }
            }
            NetworkMode::Inference => {
                for w in &mut self.weights {
                    w.clear_tape();
                }
            }
        }

        for n in &self.connections {
            (**n).borrow_mut().set_mode(mode, visited);
        }
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let connections: Vec<String> = self
            .connections
            .iter()
            .map(|n| n.borrow().id.to_string())
            .collect();
        let connections = connections.join(", ");
        write!(
            f,
            "Node: {} - connections: [{}] connected_from: {} kind: {:?}",
            self.id, connections, self.connected_from, self.kind
        )
    }
}

impl fmt::Debug for Network {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "
Network:
    - input_nodes: {}
    - nodes: {}
    - connections: {}
        ",
            self.inputs.len(),
            self.nodes.len(),
            self.get_connection_count()
        )
    }
}
