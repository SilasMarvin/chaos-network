use crate::tensors::{element_wise_mul, Tensor1D};

fn do_mish_backward(x: f64) -> f64 {
    let w = (4. * (x + 1.)) + (4. * (2. * x).exp()) + (3. * x).exp() + (x.exp() * ((4. * x) + 6.));
    let d = (2. * x.exp()) + (2. * x).exp() + 2.;
    (x.exp() * w) / d.powi(2)
}

impl<const N: usize> Tensor1D<N> {
    pub fn mish(t: &mut Self) -> Self {
        let data = t.data.map(|x| x * ((1. + x.exp()).ln()).tanh());
        let new = Tensor1D::new_with_tape(data, t.tape.clone());

        if let Some(tape) = &t.tape {
            let new_id = new.grad_for;
            let self_id = t.grad_for;
            let t_data = t.data.map(do_mish_backward);
            tape.write().add_operation((
                new_id,
                Box::new(move |g| {
                    let mut tg = g.remove(new_id);
                    tg.data = element_wise_mul::<N>(&tg.data, &t_data);
                    g.insert(self_id, tg);
                }),
            ));
        }

        new
    }

    pub fn mish_no_grad(t: &Self) -> Self {
        let data = t.data.map(|x| x * ((1. + x.exp()).ln()).tanh());
        Tensor1D::new_without_tape(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gradients::Tape, tensors::Tensor};
    use parking_lot::RwLock;
    use std::sync::Arc;

    #[test]
    fn test_mish_1d() {
        let tape: Arc<RwLock<Tape<3>>> = Arc::new(RwLock::new(Tape::new()));
        let mut a = Tensor1D::new_with_tape([1., 2., 3.], Some(tape.clone()));
        let mut b = Tensor1D::mish(&mut a);
        // Check value match
        assert_eq!(
            [0.8650983882673103, 1.9439589595339946, 2.9865350049679575],
            b.data
        );
        // Check gradients
        tape.write().checkmark_tensor_id();
        let mut grads = b.backward();
        let a_grads = grads.remove(a.id);
        assert_eq!(
            [1.0490362200997918, 1.0693179342794896, 1.0211069109294437],
            a_grads.data
        );
    }
}
