use crate::gradients::Tape;
use crate::tensors::{element_wise_mul, Tensor0D, Tensor1D, WithTape, WithoutTape};

pub trait Tensor0DMul<const N: usize, TensorTape1, TensorTape2> {
    fn mul(
        &mut self,
        left: &mut Tensor1D<N, TensorTape1>,
        tape: &mut Tape<N>,
    ) -> Tensor1D<N, TensorTape2>;

    fn mul_left_by_reference(
        &mut self,
        left: &Tensor1D<N, TensorTape1>,
        tape: &mut Tape<N>,
    ) -> Tensor1D<N, TensorTape2>;

    fn mul_explicit_no_grad(
        &self,
        left: &Tensor1D<N, TensorTape1>,
        tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithoutTape>;
}

impl<const N: usize> Tensor0DMul<N, WithoutTape, WithTape> for Tensor0D<N, WithTape> {
    fn mul(
        &mut self,
        other: &mut Tensor1D<N, WithoutTape>,
        tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithTape> {
        let new_data: [f64; N] = other.data.map(|d| self.data * d);
        let mut new = Tensor1D::new(new_data);
        new.set_id_grad_for(tape.increment_tensor_count());
        // Add operation to tape
        let new_id = new.grad_for;
        let self_id = self.grad_for;
        let other_data = other.data;
        tape.add_operation((
            new_id,
            Box::new(move |g| {
                let mut tg = g.remove(new_id);
                tg.data = element_wise_mul::<N>(&tg.data, &other_data);
                g.insert(self_id, tg);
            }),
        ));

        new
    }

    fn mul_left_by_reference(
        &mut self,
        other: &Tensor1D<N, WithoutTape>,
        tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithTape> {
        let new_data: [f64; N] = other.data.map(|d| self.data * d);
        let mut new = Tensor1D::new(new_data);
        new.set_id_grad_for(tape.increment_tensor_count());
        // Add operation to tape
        let new_id = new.grad_for;
        let self_id = self.grad_for;
        let other_data = other.data;
        tape.add_operation((
            new_id,
            Box::new(move |g| {
                let mut tg = g.remove(new_id);
                tg.data = element_wise_mul::<N>(&tg.data, &other_data);
                g.insert(self_id, tg);
            }),
        ));

        new
    }

    fn mul_explicit_no_grad(
        &self,
        other: &Tensor1D<N, WithoutTape>,
        _tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithoutTape> {
        let new_data: [f64; N] = other.data.map(|d| self.data * d);
        Tensor1D::new(new_data)
    }
}

impl<const N: usize> Tensor0DMul<N, WithTape, WithTape> for Tensor0D<N, WithTape> {
    fn mul(
        &mut self,
        other: &mut Tensor1D<N, WithTape>,
        tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithTape> {
        let new_data: [f64; N] = other.data.map(|d| self.data * d);
        let mut new = Tensor1D::new(new_data);
        new.set_id_grad_for(tape.increment_tensor_count());
        // Add operation to tape
        let new_id = new.grad_for;
        let self_id = self.grad_for;
        let other_id = other.grad_for;
        let self_data = self.data;
        let other_data = other.data;
        tape.add_operation((
            new_id,
            Box::new(move |g| {
                let mut tg1 = g.remove(new_id);
                let mut tg2 = tg1.clone();
                tg1.data = element_wise_mul::<N>(&tg1.data, &other_data);
                tg2.data = tg2.data.map(|x| x * self_data);
                g.insert(self_id, tg1);
                g.insert(other_id, tg2);
            }),
        ));

        new
    }

    fn mul_left_by_reference(
        &mut self,
        _other: &Tensor1D<N, WithTape>,
        _tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithTape> {
        todo!();
    }

    fn mul_explicit_no_grad(
        &self,
        other: &Tensor1D<N, WithTape>,
        _tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithoutTape> {
        let new_data: [f64; N] = other.data.map(|d| self.data * d);
        Tensor1D::new(new_data)
    }
}

impl<const N: usize> Tensor0DMul<N, WithoutTape, WithoutTape> for Tensor0D<N, WithoutTape> {
    fn mul(
        &mut self,
        other: &mut Tensor1D<N, WithoutTape>,
        _tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithoutTape> {
        let new_data: [f64; N] = other.data.map(|d| self.data * d);
        Tensor1D::new(new_data)
    }

    fn mul_left_by_reference(
        &mut self,
        other: &Tensor1D<N, WithoutTape>,
        _tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithoutTape> {
        let new_data: [f64; N] = other.data.map(|d| self.data * d);
        Tensor1D::new(new_data)
    }

    fn mul_explicit_no_grad(
        &self,
        other: &Tensor1D<N, WithoutTape>,
        _tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithoutTape> {
        let new_data: [f64; N] = other.data.map(|d| self.data * d);
        Tensor1D::new(new_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradients::Tape;

    #[test]
    fn test_mul_1d() {
        let mut tape: Tape<3> = Tape::new();
        let mut a: Tensor0D<3, WithTape> = Tensor0D::new(2.);
        let mut b: Tensor1D<3, WithoutTape> = Tensor1D::new([1., 2., 3.]);
        let c = a.mul(&mut b, &mut tape);
        // Check value match
        assert_eq!([2., 4., 6.], c.data);
        // Check gradients
        tape.checkmark_tensor_id();
        let mut grads = tape.execute();
        let a_grads = grads.remove(a.grad_for);
        assert_eq!([1., 2., 3.], a_grads.data);
    }
}
