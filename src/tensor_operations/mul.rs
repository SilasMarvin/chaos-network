use crate::tensors::{element_wise_mul, Tensor, Tensor0D, Tensor1D};
use std::ops::Mul;

impl<'a, 'b, const N: usize> Mul<&'b mut Tensor1D<N>> for &'a mut Tensor0D<N> {
    type Output = Tensor1D<N>;

    fn mul(self, other: &'b mut Tensor1D<N>) -> Self::Output {
        let new_data: [f64; N] = other.data.map(|d| self.data * d);
        let mut new = Tensor1D::new_without_tape(new_data);

        match (&self.tape, &other.tape) {
            (Some(self_tape), Some(_other_tape)) => {
                let new_id = new.grad_for;
                let self_id = self.grad_for;
                let other_id = other.grad_for;
                let self_data = self.data;
                let other_data = other.data.clone();
                self_tape.borrow_mut().add_operation((
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
                new.set_tape(self.tape.clone());
            }
            (Some(self_tape), None) => {
                let new_id = new.grad_for;
                let self_id = self.grad_for;
                let other_data = other.data.clone();
                self_tape.borrow_mut().add_operation((
                    new_id,
                    Box::new(move |g| {
                        let mut tg = g.remove(new_id);
                        tg.data = element_wise_mul::<N>(&tg.data, &other_data);
                        g.insert(self_id, tg);
                    }),
                ));
                new.set_tape(self.tape.clone());
            }
            (None, Some(_other_tape)) => {
                panic!("Switch operator orientation");
            }
            (None, None) => (),
        }

        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gradients::Tape, tensors::Tensor};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_mul_1d() {
        let tape: Rc<RefCell<Tape<3>>> = Rc::new(RefCell::new(Tape::new()));
        let mut a = Tensor0D::new_with_tape(2., Some(tape.clone()));
        let mut b = Tensor1D::new_without_tape([1., 2., 3.]);
        let mut c = &mut a * &mut b;
        // Check value match
        assert_eq!([2., 4., 6.], c.data);
        // Check gradients
        let mut grads = c.backward();
        let a_grads = grads.remove(a.grad_for);
        assert_eq!([1., 2., 3.], a_grads.data);
    }

    #[test]
    fn test_mul_1d_dual_grad() {
        let tape: Rc<RefCell<Tape<3>>> = Rc::new(RefCell::new(Tape::new()));
        let mut a = Tensor0D::new_with_tape(2., Some(tape.clone()));
        let mut b = Tensor1D::new_with_tape([1., 2., 3.], Some(tape.clone()));
        let mut c = &mut a * &mut b;
        // Check value match
        assert_eq!([2., 4., 6.], c.data);
        // Check gradients
        let mut grads = c.backward();
        let a_grads = grads.remove(a.grad_for);
        assert_eq!([1., 2., 3.], a_grads.data);
    }
}
