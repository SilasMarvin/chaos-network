use crate::tensors::Tensor0D;

impl Tensor0D {
    pub fn log(t: &mut Self) -> Self {
        let mut new = Tensor0D::new_without_tape(t.data.ln());

        if let Some(mut tape) = t.tape.take() {
            let new_id = new.id;
            let self_id = t.id;
            let self_data = t.data;
            tape.add_operation(Box::new(move |g| {
                let mut tg = g.remove::<Tensor0D>(new_id);
                tg.data *= 1. / self_data;
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
    use crate::tensors::Tensor;

    #[test]
    fn test_log_0d() {
        let mut a = Tensor0D::new_with_tape(3.);
        let mut b = Tensor0D::log(&mut a);
        // Check value match
        assert_eq!(b.data, 3f32.ln());
        // Check gradients
        let mut grads = b.backward();
        let a_grads = grads.remove::<Tensor0D>(a.id);
        assert_eq!(1. / 3., a_grads.data);
    }
}
