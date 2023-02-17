use crate::tensors::Tensor0D;
use std::ops::Mul;

impl<'a, 'b> Mul<&'b mut Tensor0D> for &'a mut Tensor0D {
    type Output = Tensor0D;

    fn mul(self, other: &'b mut Tensor0D) -> Self::Output {
        let mut new = Tensor0D::new_without_tape(self.data * other.data);

        new.tape = match (self.tape.take(), other.tape.take()) {
            (Some(mut self_tape), Some(other_tape)) => {
                // self_tape.merge(other_tape);
                let new_id = new.id;
                let self_id = self.id;
                let other_id = other.id;
                let self_data = self.data;
                let other_data = other.data;
                self_tape.borrow_mut().add_operation((
                    new_id,
                    Box::new(move |g| {
                        let mut tg1 = g.remove(new_id);
                        let mut tg2 = tg1.clone();
                        tg1.data *= other_data;
                        tg2.data *= self_data;
                        g.insert(self_id, tg1);
                        g.insert(other_id, tg2);
                    }),
                ));
                Some(self_tape)
            }
            (Some(mut self_tape), None) => {
                let new_id = new.id;
                let self_id = self.id;
                let other_data = other.data;
                self_tape.borrow_mut().add_operation((
                    new_id,
                    Box::new(move |g| {
                        let mut tg = g.remove(new_id);
                        tg.data *= other_data;
                        g.insert(self_id, tg);
                    }),
                ));
                Some(self_tape)
            }
            (None, Some(mut other_tape)) => {
                let new_id = new.id;
                let other_id = other.id;
                let self_data = self.data;
                other_tape.borrow_mut().add_operation((
                    new_id,
                    Box::new(move |g| {
                        let mut tg = g.remove(new_id);
                        tg.data *= self_data;
                        g.insert(other_id, tg);
                    }),
                ));
                Some(other_tape)
            }
            (None, None) => None,
        };

        new
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::tensors::Tensor;
//
//     #[test]
//     fn test_mul_0d() {
//         let mut a = Tensor0D::new_with_tape(1.);
//         let mut b = Tensor0D::new_with_tape(2.);
//         let mut c = &mut a * &mut b;
//         // Check value match
//         assert_eq!(2., c.data);
//         // Check gradients
//         let mut grads = c.backward();
//         let a_grads = grads.remove(a.id);
//         let b_grads = grads.remove(b.id);
//         assert_eq!(2., a_grads.data);
//         assert_eq!(1., b_grads.data);
//     }
// }
