use crate::tensors::Tensor0D;

impl Tensor0D {
    pub fn mish(t: &mut Self) -> Self {
        let data = t.data * ((1. + t.data.exp()).ln()).tanh();
        let mut new = Tensor0D::new_without_tape(data);

        if let Some(mut tape) = t.tape.take() {
            let new_id = new.id;
            let t_data = t.data;
            let self_id = t.id;
            tape.add_operation(Box::new(move |g| {
                let mut tg = g.remove(new_id);
                let w = (4. * (t_data + 1.))
                    + (4. * (2. * t_data).exp())
                    + (3. * t_data).exp()
                    + (t_data.exp() * ((4. * t_data) + 6.));
                let d = (2. * t_data.exp()) + (2. * t_data).exp() + 2.;
                tg.data *= (t_data.exp() * w) / d.powi(2);
                g.insert(self_id, tg);
            }));
            new.tape = Some(tape);
        }

        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensors::Tensor;

    #[test]
    fn test_mish_0d() {
        let mut a = Tensor0D::new_with_tape(1.);
        let mut b = Tensor0D::mish(&mut a);
        // Check value match
        assert_eq!(0.86509836, b.data);
        // Check gradients
        let mut grads = b.backward();
        let a_grads = grads.remove(a.id);
        assert_eq!(1.0490361, a_grads.data);
    }
}
