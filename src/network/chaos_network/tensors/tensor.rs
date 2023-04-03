use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct WithTape;
#[derive(Clone, Debug)]
pub struct WithoutTape;

#[derive(Debug, Clone)]
pub struct Tensor0D<const N: usize, Tape = WithoutTape> {
    pub id: usize,
    pub grad_for: usize,
    pub data: f64,
    tape: PhantomData<Tape>,
}

#[derive(Debug, Clone)]
pub struct Tensor1D<const N: usize, Tape = WithoutTape> {
    pub id: usize,
    pub grad_for: usize,
    pub data: [f64; N],
    tape: PhantomData<Tape>,
}

impl<const N: usize, Tape> Tensor0D<N, Tape> {
    pub fn new(data: f64) -> Self {
        Self {
            id: 0,
            grad_for: 0,
            data,
            tape: PhantomData,
        }
    }

    pub fn set_id_grad_for(&mut self, id_grad_for: usize) {
        self.id = id_grad_for;
        self.grad_for = id_grad_for;
    }
}

impl<const N: usize, Tape> Tensor1D<N, Tape> {
    pub fn new(data: [f64; N]) -> Self {
        Self {
            id: 0,
            grad_for: 0,
            data,
            tape: PhantomData,
        }
    }

    pub fn to_without_tape(self) -> Tensor1D<N, WithoutTape> {
        Tensor1D::<N, WithoutTape> {
            id: 0,
            grad_for: 0,
            data: self.data,
            tape: PhantomData,
        }
    }

    pub fn set_id_grad_for(&mut self, id_grad_for: usize) {
        self.id = id_grad_for;
        self.grad_for = id_grad_for;
    }
}
