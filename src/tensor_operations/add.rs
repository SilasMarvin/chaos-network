use crate::gradients::Tape;
use crate::tensors::{element_wise_addition, Tensor1D, WithTape, WithoutTape};

pub trait Tensor1DAdd<const N: usize, TensorTape1, TensorTape2> {
    fn add(
        &mut self,
        left: &mut Tensor1D<N, TensorTape1>,
        tape: &mut Tape<N>,
    ) -> Tensor1D<N, TensorTape2>;
}

impl<const N: usize> Tensor1DAdd<N, WithoutTape, WithTape> for Tensor1D<N, WithTape> {
    fn add(
        &mut self,
        other: &mut Tensor1D<N, WithoutTape>,
        _tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithTape> {
        let new_data = element_wise_addition(&self.data, &other.data);
        let mut new = Tensor1D::new(new_data);
        new.grad_for = self.id;
        new
    }
}

impl<const N: usize> Tensor1DAdd<N, WithTape, WithTape> for Tensor1D<N, WithTape> {
    fn add(
        &mut self,
        other: &mut Tensor1D<N, WithTape>,
        tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithTape> {
        let new_data = element_wise_addition(&self.data, &other.data);
        let mut new = Tensor1D::new(new_data);
        new.set_id_grad_for(tape.increment_tensor_count());
        // Add operation to tape
        let self_id = self.grad_for;
        let other_id = other.grad_for;
        let new_id = new.grad_for;
        tape.add_operation((
            new_id,
            Box::new(move |g| {
                let tg1 = g.remove(new_id);
                let tg2 = tg1.clone();
                g.insert(self_id, tg1);
                g.insert(other_id, tg2);
            }),
        ));

        new
    }
}

impl<const N: usize> Tensor1DAdd<N, WithoutTape, WithoutTape> for Tensor1D<N, WithoutTape> {
    fn add(
        &mut self,
        other: &mut Tensor1D<N, WithoutTape>,
        _tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithoutTape> {
        let new_data = element_wise_addition(&self.data, &other.data);
        Tensor1D::new(new_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradients::Tape;

    #[test]
    fn test_add_1d_dual_grad() {
        let mut tape: Tape<3> = Tape::new();
        let mut a: Tensor1D<3, WithTape> = Tensor1D::new([1., 2., 3.]);
        let mut b: Tensor1D<3, WithTape> = Tensor1D::new([2., 3., 4.]);
        let c = a.add(&mut b, &mut tape);
        // Check value match
        assert_eq!([3., 5., 7.], c.data);
        // Check gradients
        tape.checkmark_tensor_id();
        let mut grads = tape.execute();
        let a_grads = grads.remove(a.grad_for);
        let b_grads = grads.remove(b.grad_for);
        assert_eq!([1., 1., 1.], a_grads.data);
        assert_eq!([1., 1., 1.], b_grads.data);
    }
}
