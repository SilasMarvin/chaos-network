use crate::tensors::{Tensor, Tensor0D, Tensor1D};
use std::ops::Add;

impl<'a, 'b> Add<&'b mut Tensor0D> for &'a mut Tensor0D {
    type Output = Tensor0D;

    fn add(self, other: &'b mut Tensor0D) -> Self::Output {
        println!("ADD");
        let mut new = Tensor0D::new_without_tape(self.data + other.data);
        new.tape = match (&self.tape, &other.tape) {
            (Some(self_tape), Some(_other_tape)) => {
                let new_id = new.grad_for;
                let self_id = self.grad_for;
                let other_id = other.grad_for;
                self_tape.borrow_mut().add_operation((
                    new_id,
                    Box::new(move |g| {
                        let tg1 = g.remove(new_id);
                        let tg2 = tg1.clone();
                        println!("Add Insert: {} {}", tg1.data, tg2.data);
                        g.insert(self_id, tg1);
                        g.insert(other_id, tg2);
                    }),
                ));
                Some(self_tape.clone())
            }
            (Some(self_tape), None) => {
                println!("ADD grad_for");
                new.grad_for = self.id;
                Some(self_tape.clone())
            }
            (None, Some(other_tape)) => {
                println!("ADD grad_for");
                new.grad_for = other.id;
                Some(other_tape.clone())
            }
            (None, None) => None,
        };

        new
    }
}

impl<'a, 'b, const N: usize> Add<&'b mut Tensor1D<N>> for &'a mut Tensor1D<N> {
    type Output = Tensor1D<N>;

    fn add(self, other: &'b mut Tensor1D<N>) -> Self::Output {
        println!("ADD");
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
                self_tape.borrow_mut().add_operation((
                    new_id,
                    Box::new(move |g| {
                        let tg1 = g.remove(new_id);
                        let tg2 = tg1.clone();
                        println!("Add Insert: {} {}", tg1.data, tg2.data);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gradients::Tape, tensors::Tensor};
    use std::cell::RefCell;
    use std::rc::Rc;

    // #[test]
    // fn test_add_0d() {
    //     let tape = Rc::new(RefCell::new(Tape::new()));
    //     let mut a = Tensor0D::new_with_tape(1., Some(tape.clone()));
    //     let mut b = Tensor0D::new_with_tape(2., Some(tape.clone()));
    //     let mut c = &mut a + &mut b;
    //     // Check value match
    //     assert_eq!(3., c.data);
    //     // Check gradients
    //     let mut grads = c.backward();
    //     let a_grads = grads.remove(a.id);
    //     let b_grads = grads.remove(b.id);
    //     assert_eq!(1., a_grads.data);
    //     assert_eq!(1., b_grads.data);
    // }

    // #[test]
    // fn test_add_1d_dual_grad() {
    //     let tape = Rc::new(RefCell::new(Tape::new()));
    //     let mut a = Tensor0D::new_with_tape(2., Some(tape.clone()));
    //     let mut b = Tensor1D::new_without_tape([1., 2., 3.]);
    //     let mut c = Tensor0D::new_with_tape(3., Some(tape.clone()));
    //     let mut d = Tensor1D::new_without_tape([2., 3., 4.]);
    //     let mut e = Tensor0D::new_with_tape(4., Some(tape.clone()));
    //     let mut f = &mut a * &mut b;
    //     let mut g = &mut c * &mut d;
    //     let mut h = &mut f + &mut g;
    //     let mut i = &mut e * &mut h;
    //     // Check value match
    //     assert_eq!([32., 52., 72.], i.data);
    //     // Check gradients
    //     let mut grads = i.backward();
    //     let a_grads = grads.remove(a.grad_for);
    //     let c_grads = grads.remove(c.grad_for);
    //     let e_grads = grads.remove(e.grad_for);
    //     assert_eq!(24., a_grads.data);
    //     assert_eq!(36., c_grads.data);
    //     assert_eq!(39., e_grads.data);
    // }
}
