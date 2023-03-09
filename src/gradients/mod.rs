use crate::tensors::{Tensor, Tensor1D};
use rustc_hash::FxHashMap;

#[derive(Default)]
pub struct Tape<const N: usize> {
    operations: Vec<(usize, Box<dyn FnOnce(&mut Gradients<N>) + Send + Sync>)>,
    current_tensor_id: usize,
    checkmarked_tensor_id: usize,
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
            current_tensor_id: 0,
            checkmarked_tensor_id: 0,
        }
    }

    pub fn add_operation(
        &mut self,
        operation: (usize, Box<dyn FnOnce(&mut Gradients<N>) + Send + Sync>),
    ) {
        self.operations.push(operation)
    }

    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    pub fn checkmark_tensor_id(&mut self) {
        self.checkmarked_tensor_id = self.current_tensor_id;
    }

    pub fn execute(&mut self) -> Gradients<N> {
        let mut gradients = Gradients::default();
        self.operations.sort_by(|a, b| a.0.cmp(&b.0));
        for operation in self.operations.drain(..).rev() {
            (operation.1)(&mut gradients);
        }
        self.current_tensor_id = self.checkmarked_tensor_id;
        gradients
    }

    pub fn register_and_set_id(&mut self) -> usize {
        self.current_tensor_id += 1;
        self.current_tensor_id
    }
}

#[derive(Debug)]
pub struct Gradients<const N: usize> {
    pub grads: FxHashMap<usize, Tensor1D<N>>,
}

impl<const N: usize> Gradients<N> {
    pub fn remove(&mut self, id: usize) -> Tensor1D<N> {
        self.grads
            .remove(&id)
            .unwrap_or(Tensor1D::default_without_tape())
    }

    pub fn remove_or_0(&mut self, id: usize) -> Tensor1D<N> {
        self.grads
            .remove(&id)
            .unwrap_or(Tensor1D::new_without_tape([0.; N]))
    }

    pub fn insert(&mut self, key: usize, tensor: Tensor1D<N>) {
        self.grads.insert(key, tensor);
    }
}

impl<const N: usize> Default for Gradients<N> {
    fn default() -> Self {
        Self {
            grads: FxHashMap::default(),
        }
    }
}
