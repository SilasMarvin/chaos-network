use rustc_hash::FxHashMap;

use crate::tensors::{Tensor, Tensor1D};

// fn translate_range(x: u64, allowed_min: u64, allowed_max: u64, min: u64, max: u64) -> u64 {
//     (allowed_max) * (x - min) / (max - min)
// }

#[derive(Default)]
pub struct Tape<const N: usize> {
    operations: Vec<(u64, Box<dyn FnOnce(&mut Gradients<N>) + Send + Sync>)>,
}

impl<const N: usize> std::fmt::Debug for Tape<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientTape")
            .field("num_operations", &self.operations.len())
            .finish()
    }
}

impl<const N: usize> Tape<N> {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn add_operation(
        &mut self,
        operation: (u64, Box<dyn FnOnce(&mut Gradients<N>) + Send + Sync>),
    ) {
        self.operations.push(operation)
    }

    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    pub fn execute(&mut self) -> Gradients<N> {
        let mut gradients: Gradients<N> = Gradients::default();
        self.operations.sort_by(|a, b| a.0.cmp(&b.0));
        // gradients.min = self.operations.first().unwrap().0;
        // gradients.max = self.operations.last().unwrap().0;
        for operation in self.operations.drain(..).rev() {
            (operation.1)(&mut gradients);
        }
        gradients
    }
}

#[derive(Debug)]
pub struct Gradients<const N: usize> {
    pub grads: FxHashMap<u64, Tensor1D<N>>,
    // min: u64,
    // max: u64,
}

impl<const N: usize> Gradients<N> {
    pub fn remove(&mut self, id: u64) -> Tensor1D<N> {
        self.grads
            .remove(&id)
            .unwrap_or(Tensor1D::default_without_tape())
    }

    pub fn remove_or_0(&mut self, id: u64) -> Tensor1D<N> {
        self.grads
            .remove(&id)
            .unwrap_or(Tensor1D::new_without_tape([0.; N]))
    }

    pub fn insert(&mut self, key: u64, tensor: Tensor1D<N>) {
        self.grads.insert(key, tensor);
    }
}

impl<const N: usize> Default for Gradients<N> {
    fn default() -> Self {
        Self {
            grads: FxHashMap::default(),
            // min: 0,
            // max: 0,
        }
    }
}
