use crate::tensors::Tensor1D;
use std::ops::Add;

impl<'a, 'b, const N: usize> Add<&'b mut Tensor1D<N>> for &'a mut Tensor1D<N> {
    type Output = Tensor1D<N>;

    fn add(self, other: &'b mut Tensor1D<N>) -> Self::Output {
        let mut tracker = 0;
        let new_data: [f64; N] = self.data.map(|a| {
            let x = a + other.data[tracker];
            tracker += 1;
            x
        });
        let mut new = Tensor1D::new_without_tape(new_data);
        new.tape = match (&self.tape, &other.tape) {
            (Some(self_tape), Some(_other_tape)) => {
                let new_id = new.grad_for;
                let self_id = self.grad_for;
                let other_id = other.grad_for;
                self_tape.write().unwrap().add_operation((
                    new_id,
                    Box::new(move |g| {
                        let tg1 = g.remove(new_id);
                        let tg2 = tg1.clone();
                        g.insert(self_id, tg1);
                        g.insert(other_id, tg2);
                    }),
                ));
                self.tape.clone()
            }
            (Some(_self_tape), None) => {
                new.grad_for = self.id;
                self.tape.clone()
            }
            (None, Some(_other_tape)) => {
                new.grad_for = other.id;
                other.tape.clone()
            }
            (None, None) => None,
        };

        new
    }
}

impl<'a, 'b, const N: usize> Add<&'b Tensor1D<N>> for &'a Tensor1D<N> {
    type Output = Tensor1D<N>;

    fn add(self, other: &'b Tensor1D<N>) -> Self::Output {
        let mut tracker = 0;
        let new_data: [f64; N] = self.data.map(|a| {
            let x = a + other.data[tracker];
            tracker += 1;
            x
        });
        Tensor1D::new_without_tape(new_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gradients::Tape, tensors::Tensor};
    use std::sync::Arc;
    use std::sync::RwLock;

    #[test]
    fn test_add_1d_dual_grad() {
        let tape: Arc<RwLock<Tape<3>>> = Arc::new(RwLock::new(Tape::new()));
        let mut a = Tensor1D::new_with_tape([1., 2., 3.], Some(tape.clone()));
        let mut b = Tensor1D::new_with_tape([2., 3., 4.], Some(tape.clone()));
        let mut c = &mut a + &mut b;
        // Check value match
        assert_eq!([3., 5., 7.], c.data);
        // Check gradients
        let mut grads = c.backward();
        let a_grads = grads.remove(a.grad_for);
        let b_grads = grads.remove(b.grad_for);
        assert_eq!([1., 1., 1.], a_grads.data);
        assert_eq!([1., 1., 1.], b_grads.data);
    }
}
