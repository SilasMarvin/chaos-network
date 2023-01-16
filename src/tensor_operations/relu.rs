use crate::tensors::{Tensor, Tensor0D};

impl Tensor0D {
    pub fn relu(t: &mut Self) -> Self {
        let data = t.data.max(0.);
        let mut new = Tensor0D::new_without_tape(data);

        if let Some(mut tape) = t.tape.take() {
            let new_id = new.id;
            let self_id = t.id;
            tape.add_operation(Box::new(move |g| {
                let mut tg = g.remove::<Tensor0D>(new_id);
                if data == 0. {
                    tg.data = 0.;
                }
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
        let mut a = Tensor0D::new_with_tape(-1.);
        let mut b = Tensor0D::new_with_tape(1.);
        // Check value match
        let mut c = Tensor0D::relu(&mut a);
        let mut d = Tensor0D::relu(&mut b);
        assert_eq!(0., c.data);
        assert_eq!(1., d.data);
        // Check gradients
        let mut grads = c.backward();
        let a_grads = grads.remove::<Tensor0D>(a.id);
        assert_eq!(0., a_grads.data);
        let mut grads = d.backward();
        let b_grads = grads.remove::<Tensor0D>(b.id);
        assert_eq!(1., b_grads.data);
    }
}
