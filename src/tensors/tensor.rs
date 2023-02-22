use std::cell::RefCell;
use std::rc::Rc;

use crate::gradients::{Gradients, Tape};

pub trait Tensor {
    fn default_without_tape() -> Self;
    fn backward(&mut self) -> Gradients;
    fn set_tape(&mut self, tape: Option<Rc<RefCell<Tape>>>);
    fn clear_tape(&mut self);
    // fn add(self, other: Self) -> Self;
    // fn sub(self, other: Self) -> Self;
    // // fn add_scalar(self, other: f64) -> Self;
    // fn sub_scalar(self, other: f64) -> Self;
    // fn mul(self, other: Self) -> Self;
    // fn mul_scalar(self, other: f64) -> Self;
    // fn square(self) -> Self;
    // fn dot(self, other: Self) -> Tensor0D;
}

#[derive(Debug, Clone)]
pub struct Tensor0D {
    pub id: u64,
    pub grad_for: u64,
    pub data: f64,
    pub tape: Option<Rc<RefCell<Tape>>>,
}

#[derive(Debug, Clone)]
pub struct Tensor1D<const N: usize> {
    pub id: u64,
    pub grad_for: u64,
    pub data: [f64; N],
    pub tape: Option<Rc<RefCell<Tape>>>,
}

impl Tensor for Tensor0D {
    fn default_without_tape() -> Self {
        Tensor0D::new_without_tape(1.)
    }

    fn backward(&mut self) -> Gradients {
        match &mut self.tape.take() {
            Some(tape) => tape.borrow_mut().execute(),
            None => Gradients::default(),
        }
    }

    fn set_tape(&mut self, tape: Option<Rc<RefCell<Tape>>>) {
        self.tape = tape;
    }

    fn clear_tape(&mut self) {
        self.tape = None;
    }
}

impl<const N: usize> Tensor for Tensor1D<N> {
    fn default_without_tape() -> Self {
        Self::new_without_tape([1.; N])
    }

    fn backward(&mut self) -> Gradients {
        match &mut self.tape.take() {
            Some(tape) => tape.borrow_mut().execute(),
            None => Gradients::default(),
        }
    }

    fn set_tape(&mut self, tape: Option<Rc<RefCell<Tape>>>) {
        self.tape = tape;
    }

    fn clear_tape(&mut self) {
        self.tape = None;
    }
}
