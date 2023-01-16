use crate::tensors::{Tensor, Tensor0D};

impl Tensor0D {
    pub fn mse(t: &mut Self, other: &mut Self) -> Self {
        let data = (t.data - other.data).powi(2);
        let mut new = Tensor0D::new_without_tape(data);

        if let Some(mut tape) = t.tape.take() {
            let self_id = t.id;
            let self_data = t.data;
            let other_data = other.data;
            tape.add_operation(Box::new(move |g| {
                let mut tg = g.remove::<Tensor0D>(self_id);
                tg.data *= 2. * (self_data - other_data);
                g.insert::<Tensor0D>(self_id, tg);
            }));
            new.tape = Some(tape);
        }

        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_0d() {
        let mut a = Tensor0D::new_with_tape(1.);
        let mut b = Tensor0D::new_without_tape(10.);
        // Check value match
        let mut c = Tensor0D::mse(&mut a, &mut b);
        assert_eq!(81., c.data);
        // Check gradients
        let mut grads = c.backward();
        let a_grads = grads.remove::<Tensor0D>(a.id);
        assert_eq!(-18., a_grads.data);
    }
}
