use std::collections::HashMap;

use crate::tensors::{Tensor, Tensor0D};

#[derive(Default)]
pub struct Tape {
    operations: Vec<Box<dyn FnOnce(&mut Gradients)>>,
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

    pub fn add_operation(&mut self, operation: Box<dyn FnOnce(&mut Gradients)>) {
        self.operations.push(operation)
    }

    pub fn merge(&mut self, mut other: Self) {
        other.operations.append(&mut self.operations);
        self.operations = other.operations;
    }

    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
}

impl Tape {
    pub fn execute(&mut self) -> Gradients {
        let mut gradients: Gradients = Gradients::default();
        for operation in self.operations.drain(..).rev() {
            (operation)(&mut gradients);
        }
        gradients
    }
}

#[derive(Debug, Default)]
pub struct Gradients {
    grads: HashMap<i32, Tensor0D>,
}

impl Gradients {
    pub fn remove(&mut self, id: i32) -> Tensor0D {
        self.grads
            .remove(&id)
            .unwrap_or(Tensor0D::default_without_tape())
    }

    pub fn insert(&mut self, key: i32, tensor: Tensor0D) {
        self.grads.insert(key, tensor);
    }
}
