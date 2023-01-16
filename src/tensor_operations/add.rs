use crate::tensors::{Tensor, Tensor0D};
use std::ops::Add;

impl<'a, 'b> Add<&'b mut Tensor0D> for &'a mut Tensor0D {
    type Output = Tensor0D;

    fn add(self, other: &'b mut Tensor0D) -> Self::Output {
        let mut new = Tensor0D::new_without_tape(self.data + other.data);

        new.tape = match (self.tape.take(), other.tape.take()) {
            (Some(mut self_tape), Some(other_tape)) => {
                self_tape.merge(other_tape);
                let new_id = new.id;
                let self_id = self.id;
                let other_id = other.id;
                self_tape.add_operation(Box::new(move |g| {
                    let tg1 = g.remove::<Tensor0D>(new_id);
                    let tg2 = tg1.clone();
                    g.insert::<Tensor0D>(self_id, tg1);
                    g.insert::<Tensor0D>(other_id, tg2);
                }));
                Some(self_tape)
            }
            (Some(mut self_tape), None) => {
                let new_id = new.id;
                let self_id = self.id;
                self_tape.add_operation(Box::new(move |g| {
                    let tg = g.remove::<Tensor0D>(new_id);
                    g.insert::<Tensor0D>(self_id, tg);
                }));
                Some(self_tape)
            }
            (None, Some(mut other_tape)) => {
                let new_id = new.id;
                let other_id = other.id;
                other_tape.add_operation(Box::new(move |g| {
                    let tg = g.remove::<Tensor0D>(new_id);
                    g.insert::<Tensor0D>(other_id, tg);
                }));
                Some(other_tape)
            }
            (None, None) => None,
        };

        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_0d() {
        let mut a = Tensor0D::new_with_tape(1.);
        let mut b = Tensor0D::new_with_tape(2.);
        let mut c = &mut a + &mut b;
        // Check value match
        assert_eq!(3., c.data);
        // Check gradients
        let mut grads = c.backward();
        let a_grads = grads.remove::<Tensor0D>(a.id);
        let b_grads = grads.remove::<Tensor0D>(b.id);
        assert_eq!(1., a_grads.data);
        assert_eq!(1., b_grads.data);
    }
}
