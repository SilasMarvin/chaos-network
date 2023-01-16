use crate::gradients::{Gradients, Tape};

pub trait Tensor {
    fn default_without_tape() -> Self;
    fn backward(&mut self) -> Gradients;
    fn reset_tape(&mut self);
    fn clear_tape(&mut self);
    // fn add(self, other: Self) -> Self;
    // fn sub(self, other: Self) -> Self;
    // // fn add_scalar(self, other: f32) -> Self;
    // fn sub_scalar(self, other: f32) -> Self;
    // fn mul(self, other: Self) -> Self;
    // fn mul_scalar(self, other: f32) -> Self;
    // fn square(self) -> Self;
    // fn dot(self, other: Self) -> Tensor0D;
}

#[derive(Debug)]
pub struct Tensor0D {
    pub id: i32,
    pub data: f32,
    pub tape: Option<Tape>,
}

impl Tensor for Tensor0D {
    fn default_without_tape() -> Self {
        Tensor0D::new_without_tape(1.)
    }

    fn backward(&mut self) -> Gradients {
        match &mut self.tape.take() {
            Some(tape) => tape.execute(),
            None => Gradients::default(),
        }
    }

    fn reset_tape(&mut self) {
        self.tape = Some(Tape::new());
    }

    fn clear_tape(&mut self) {
        self.tape = None;
    }
}
