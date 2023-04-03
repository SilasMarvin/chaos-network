use crate::network::chaos_network::tensors::Tensor1D;

#[derive(Default)]
pub struct Tape<const N: usize> {
    operations: Vec<(usize, Box<dyn FnOnce(&mut Gradients<N>) + Send + Sync>)>,
    current_network_tensor_id: usize,
    removed_network_tensor_ids: Vec<usize>,
    current_temporary_tensor_id: usize,
    gradients: Gradients<N>,
}

impl<const N: usize> std::fmt::Debug for Tape<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientTape")
            .field("num_operations", &self.operations.len())
            .finish()
    }
}

impl<const N: usize> Clone for Tape<N> {
    fn clone(&self) -> Self {
        Self {
            operations: Vec::new(),
            current_network_tensor_id: self.current_network_tensor_id,
            removed_network_tensor_ids: self.removed_network_tensor_ids.clone(),
            current_temporary_tensor_id: self.current_temporary_tensor_id,
            gradients: Gradients::default(),
        }
    }
}

impl<const N: usize> Tape<N> {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            current_network_tensor_id: 0,
            removed_network_tensor_ids: Vec::new(),
            current_temporary_tensor_id: 0,
            gradients: Gradients::default(),
        }
    }

    pub fn add_operation(
        &mut self,
        operation: (usize, Box<dyn FnOnce(&mut Gradients<N>) + Send + Sync>),
    ) {
        self.operations.push(operation)
    }

    pub fn execute(&mut self) -> &mut Gradients<N> {
        self.operations.sort_by(|a, b| a.0.cmp(&b.0));
        if self.gradients.grads.len()
            < self.current_network_tensor_id + self.current_temporary_tensor_id + 1
        {
            self.gradients.grads.resize(
                self.current_network_tensor_id + self.current_temporary_tensor_id + 1,
                None,
            );
        }
        for operation in self.operations.drain(..).rev() {
            (operation.1)(&mut self.gradients);
        }
        // We are assuming the only tensors with tape associated with this tape are network tensors
        self.current_temporary_tensor_id = 0;
        &mut self.gradients
    }

    pub fn get_next_network_tensor_id(&mut self) -> usize {
        if self.removed_network_tensor_ids.len() > 0 {
            self.removed_network_tensor_ids.pop().unwrap()
        } else {
            self.current_network_tensor_id += 1;
            self.current_network_tensor_id
        }
    }

    pub fn remove_network_tensor(&mut self, id: usize) {
        self.removed_network_tensor_ids.push(id);
    }

    pub fn get_next_temporary_tensor_id(&mut self) -> usize {
        self.current_temporary_tensor_id += 1;
        self.current_temporary_tensor_id + self.current_network_tensor_id
    }
}

#[derive(Debug)]
pub struct Gradients<const N: usize> {
    pub grads: Vec<Option<Tensor1D<N>>>,
}

impl<const N: usize> Gradients<N> {
    pub fn remove(&mut self, id: usize) -> Tensor1D<N> {
        let x = std::mem::take(&mut self.grads[id]);
        match x {
            Some(t) => t,
            None => Tensor1D::new([1.; N]),
        }
    }

    pub fn remove_or_0(&mut self, id: usize) -> Tensor1D<N> {
        let x = std::mem::take(&mut self.grads[id]);
        match x {
            Some(t) => t,
            None => Tensor1D::new([0.; N]),
        }
    }

    // pub fn remove_raw(&mut self, id: usize) -> Option<Tensor1D<N>> {
    //     std::mem::take(&mut self.grads[id])
    // }

    pub fn insert(&mut self, key: usize, tensor: Tensor1D<N>) {
        // Super fast insert
        unsafe {
            *self.grads.get_unchecked_mut(key) = Some(tensor);
        }
    }
}

impl<const N: usize> Default for Gradients<N> {
    fn default() -> Self {
        Self { grads: Vec::new() }
    }
}

// use crate::tensors::Tensor1D;
// use rustc_hash::FxHashMap;
//
// #[derive(Default)]
// pub struct Tape<const N: usize> {
//     operations: Vec<(usize, Box<dyn FnOnce(&mut Gradients<N>) + Send + Sync>)>,
//     current_tensor_id: usize,
//     checkmarked_tensor_id: usize,
// }
//
// impl<const N: usize> std::fmt::Debug for Tape<N> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("GradientTape")
//             .field("num_operations", &self.operations.len())
//             .finish()
//     }
// }
//
// impl<const N: usize> Tape<N> {
//     pub fn new() -> Self {
//         Self {
//             operations: Vec::new(),
//             current_tensor_id: 0,
//             checkmarked_tensor_id: 0,
//         }
//     }
//
//     pub fn add_operation(
//         &mut self,
//         operation: (usize, Box<dyn FnOnce(&mut Gradients<N>) + Send + Sync>),
//     ) {
//         self.operations.push(operation)
//     }
//
//     pub fn checkmark_tensor_id(&mut self) {
//         self.checkmarked_tensor_id = self.current_tensor_id;
//     }
//
//     pub fn execute(&mut self) -> Gradients<N> {
//         let mut gradients = Gradients::default();
//         self.operations.sort_by(|a, b| a.0.cmp(&b.0));
//         for operation in self.operations.drain(..).rev() {
//             (operation.1)(&mut gradients);
//         }
//         self.current_tensor_id = self.checkmarked_tensor_id;
//         gradients
//     }
//
//     pub fn get_next_temporary_tensor_id(&mut self) -> usize {
//         self.current_tensor_id += 1;
//         self.current_tensor_id
//     }
// }
//
// #[derive(Debug)]
// pub struct Gradients<const N: usize> {
//     pub grads: FxHashMap<usize, Tensor1D<N>>,
// }
//
// impl<const N: usize> Gradients<N> {
//     pub fn remove(&mut self, id: usize) -> Tensor1D<N> {
//         self.grads.remove(&id).unwrap_or(Tensor1D::new([1.; N]))
//     }
//
//     pub fn remove_or_0(&mut self, id: usize) -> Tensor1D<N> {
//         self.grads.remove(&id).unwrap_or(Tensor1D::new([0.; N]))
//     }
//
//     pub fn insert(&mut self, key: usize, tensor: Tensor1D<N>) {
//         self.grads.insert(key, tensor);
//     }
// }
//
// impl<const N: usize> Default for Gradients<N> {
//     fn default() -> Self {
//         Self {
//             grads: FxHashMap::default(),
//         }
//     }
// }

// use crate::tensors::{Tensor, Tensor1D};
// use rustc_hash::FxHashMap;
//
// #[derive(Default)]
// pub struct Tape<const N: usize> {
//     operations: Vec<(usize, Box<dyn FnOnce(&mut Gradients<N>) + Send + Sync>)>,
//     current_tensor_id: usize,
//     checkmarked_tensor_id: usize,
// }
//
// impl<const N: usize> std::fmt::Debug for Tape<N> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("GradientTape")
//             .field("num_operations", &self.operations.len())
//             .finish()
//     }
// }
//
// impl<const N: usize> Tape<N> {
//     pub fn new() -> Self {
//         Self {
//             operations: Vec::new(),
//             current_tensor_id: 0,
//             checkmarked_tensor_id: 0,
//         }
//     }
//
//     pub fn add_operation(
//         &mut self,
//         operation: (usize, Box<dyn FnOnce(&mut Gradients<N>) + Send + Sync>),
//     ) {
//         self.operations.push(operation)
//     }
//
//     pub fn operation_count(&self) -> usize {
//         self.operations.len()
//     }
//
//     pub fn checkmark_tensor_id(&mut self) {
//         self.checkmarked_tensor_id = self.current_tensor_id;
//     }
//
//     pub fn execute(&mut self) -> Gradients<N> {
//         let mut gradients = Gradients::default();
//         self.operations.sort_by(|a, b| a.0.cmp(&b.0));
//         for operation in self.operations.drain(..).rev() {
//             (operation.1)(&mut gradients);
//         }
//         self.current_tensor_id = self.checkmarked_tensor_id;
//         gradients
//     }
//
//     pub fn register_and_set_id(&mut self) -> usize {
//         self.current_tensor_id += 1;
//         self.current_tensor_id
//     }
// }
//
// #[derive(Debug)]
// pub struct Gradients<const N: usize> {
//     pub grads: FxHashMap<usize, Tensor1D<N>>,
// }
//
// impl<const N: usize> Gradients<N> {
//     pub fn remove(&mut self, id: usize) -> Tensor1D<N> {
//         self.grads
//             .remove(&id)
//             .unwrap_or(Tensor1D::default_without_tape())
//     }
//
//     pub fn remove_or_0(&mut self, id: usize) -> Tensor1D<N> {
//         self.grads
//             .remove(&id)
//             .unwrap_or(Tensor1D::new_without_tape([0.; N]))
//     }
//
//     pub fn insert(&mut self, key: usize, tensor: Tensor1D<N>) {
//         self.grads.insert(key, tensor);
//     }
// }
//
// impl<const N: usize> Default for Gradients<N> {
//     fn default() -> Self {
//         Self {
//             grads: FxHashMap::default(),
//         }
//     }
// }
