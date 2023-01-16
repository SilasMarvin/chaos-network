use std::any::Any;
use std::collections::HashMap;

use crate::tensors::Tensor;

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

#[derive(Debug)]
pub struct Gradients {
    grads: HashMap<i32, Box<dyn Any>>,
}

impl Gradients {
    pub fn default() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    pub fn remove<T: Tensor + 'static>(&mut self, id: i32) -> Box<T> {
        let it: Box<dyn Any> = self
            .grads
            .remove(&id)
            .or(Some(Box::new(T::default_without_tape()) as Box<dyn Any>))
            .unwrap();
        match it.downcast::<T>() {
            Ok(i) => i,
            Err(_e) => panic!("Could not downcast"),
        }
    }

    pub fn insert<T: Tensor>(&mut self, key: i32, tensor: Box<dyn Any>) {
        self.grads.insert(key, tensor);
    }
}
