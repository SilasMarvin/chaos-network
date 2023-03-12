use crate::gradients::Tape;
use crate::tensors::{element_wise_mul, Tensor1D, WithTape, WithoutTape};

fn do_mish_backward(x: f64) -> f64 {
    let w = (4. * (x + 1.)) + (4. * (2. * x).exp()) + (3. * x).exp() + (x.exp() * ((4. * x) + 6.));
    let d = (2. * x.exp()) + (2. * x).exp() + 2.;
    (x.exp() * w) / d.powi(2)
}

pub trait Tensor1DMish<const N: usize, TensorTape1> {
    fn mish(&mut self, tape: &mut Tape<N>) -> Tensor1D<N, TensorTape1>;
}

impl<const N: usize> Tensor1DMish<N, WithTape> for Tensor1D<N, WithTape> {
    fn mish(&mut self, tape: &mut Tape<N>) -> Tensor1D<N, WithTape> {
        let data = self.data.map(|x| x * ((1. + x.exp()).ln()).tanh());
        let mut new = Tensor1D::new(data);
        new.set_id_grad_for(tape.get_next_temporary_tensor_id());

        // Add operation to tape
        let new_id = new.grad_for;
        let self_id = self.grad_for;
        let t_data = self.data.map(do_mish_backward);
        tape.add_operation((
            new_id,
            Box::new(move |g| {
                let mut tg = g.remove(new_id);
                tg.data = element_wise_mul::<N>(&tg.data, &t_data);
                g.insert(self_id, tg);
            }),
        ));

        new
    }
}

impl<const N: usize> Tensor1DMish<N, WithoutTape> for Tensor1D<N, WithoutTape> {
    fn mish(&mut self, _tape: &mut Tape<N>) -> Tensor1D<N, WithoutTape> {
        let data = self.data.map(|x| x * ((1. + x.exp()).ln()).tanh());
        Tensor1D::new(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradients::Tape;

    #[test]
    fn test_mish_1d() {
        let mut tape: Tape<3> = Tape::new();
        let mut a: Tensor1D<3, WithTape> = Tensor1D::new([1., 2., 3.]);
        let b = a.mish(&mut tape);
        // Check value match
        assert_eq!(
            [0.8650983882673103, 1.9439589595339946, 2.9865350049679575],
            b.data
        );
        // Check gradients
        let grads = tape.execute();
        let a_grads = grads.remove(a.id);
        assert_eq!(
            [1.0490362200997918, 1.0693179342794896, 1.0211069109294437],
            a_grads.data
        );
    }
}
