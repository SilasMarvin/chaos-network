use fasthash::xx::Hash64;
use fasthash::{city, spooky, xx, RandomState, XXHasher};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use crate::tensors::{Tensor, Tensor0D};

const WORKERS: usize = 32;

#[derive(Default)]
pub struct Tape {
    operations: Vec<(u64, Box<dyn FnOnce(&mut Gradients) + Send + Sync>)>,
}

impl std::fmt::Debug for Tape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientTape")
            .field("num_operations", &self.operations.len())
            .finish()
    }
}

impl Tape {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn add_operation(
        &mut self,
        operation: (u64, Box<dyn FnOnce(&mut Gradients) + Send + Sync>),
    ) {
        self.operations.push(operation)
    }

    pub fn merge(&mut self, mut other: Self) {
        other.operations.append(&mut self.operations);
        self.operations = other.operations;
    }

    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    pub fn execute(&mut self) -> Gradients {
        let mut gradients: Gradients = Gradients::default();
        self.operations.sort_by(|a, b| a.0.cmp(&b.0));
        // println!("Operations: {:?}", self.operations);
        for operation in self.operations.drain(..).rev() {
            (operation.1)(&mut gradients);
        }
        gradients
    }
}

#[derive(Debug)]
pub struct Gradients {
    grads: HashMap<u64, Tensor0D, xx::Hash32>,
}

impl Gradients {
    pub fn remove(&mut self, id: u64) -> Tensor0D {
        self.grads
            .remove(&id)
            .unwrap_or(Tensor0D::default_without_tape())
    }

    pub fn remove_or_0(&mut self, id: u64) -> Tensor0D {
        self.grads
            .remove(&id)
            .unwrap_or(Tensor0D::new_without_tape(0.))
    }

    pub fn insert(&mut self, key: u64, tensor: Tensor0D) {
        self.grads.insert(key, tensor);
    }
}

impl Default for Gradients {
    fn default() -> Self {
        Self {
            grads: HashMap::with_hasher(xx::Hash32),
        }
    }
}
