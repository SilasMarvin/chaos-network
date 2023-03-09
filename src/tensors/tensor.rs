use parking_lot::RwLock;
use std::sync::Arc;

use crate::gradients::{Gradients, Tape};

pub trait Tensor<const N: usize> {
    fn default_without_tape() -> Self;
    fn backward(&mut self) -> Gradients<N>;
    fn set_tape(&mut self, tape: Option<Arc<RwLock<Tape<N>>>>);
    fn set_tape_no_id(&mut self, tape: Option<Arc<RwLock<Tape<N>>>>);
    fn clear_tape(&mut self);
}

#[derive(Debug, Clone)]
pub struct Tensor0D<const N: usize> {
    pub id: usize,
    pub grad_for: usize,
    pub data: f64,
    pub tape: Option<Arc<RwLock<Tape<N>>>>,
}

#[derive(Debug, Clone)]
pub struct Tensor1D<const N: usize> {
    pub id: usize,
    pub grad_for: usize,
    pub data: [f64; N],
    pub tape: Option<Arc<RwLock<Tape<N>>>>,
}

impl<const N: usize> Tensor<N> for Tensor0D<N> {
    fn default_without_tape() -> Self {
        Tensor0D::new_without_tape(1.)
    }

    fn backward(&mut self) -> Gradients<N> {
        match &mut self.tape.take() {
            Some(tape) => tape.write().execute(),
            None => panic!("Calling backwards on a tensor that does not have a tape"),
        }
        // match &mut self.tape.take() {
        //     Some(tape) => match tape.write() {
        //         Ok(mut t) => t.execute(),
        //         Err(_e) => panic!("Error unwrapping tape"),
        //     },
        //     None => panic!("Calling backwards on a tensor that does not have a tape"),
        // }
    }

    fn set_tape(&mut self, tape: Option<Arc<RwLock<Tape<N>>>>) {
        match tape {
            Some(t) => {
                let id_grad_for = t.write().register_and_set_id();
                self.id = id_grad_for;
                self.grad_for = id_grad_for;
                self.tape = Some(t)
            }
            None => {
                self.tape = None;
                self.id = 0;
            }
        }
    }

    fn set_tape_no_id(&mut self, tape: Option<Arc<RwLock<Tape<N>>>>) {
        self.tape = tape;
    }

    fn clear_tape(&mut self) {
        self.tape = None;
    }
}

impl<const N: usize> Tensor<N> for Tensor1D<N> {
    fn default_without_tape() -> Self {
        Self::new_without_tape([1.; N])
    }

    fn backward(&mut self) -> Gradients<N> {
        match &mut self.tape.take() {
            Some(tape) => tape.write().execute(),
            None => panic!("Calling backwards on a tensor that does not have a tape"),
        }
    }

    fn set_tape(&mut self, tape: Option<Arc<RwLock<Tape<N>>>>) {
        match tape {
            Some(t) => {
                let id_grad_for = t.write().register_and_set_id();
                self.id = id_grad_for;
                self.grad_for = id_grad_for;
                self.tape = Some(t)
            }
            None => {
                self.tape = None;
                self.id = 0;
            }
        }
    }

    fn set_tape_no_id(&mut self, tape: Option<Arc<RwLock<Tape<N>>>>) {
        self.tape = tape;
    }

    fn clear_tape(&mut self) {
        self.tape = None;
    }
}
