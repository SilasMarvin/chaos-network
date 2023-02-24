use crate::tensors::{element_wise_mul, Tensor, Tensor0D, Tensor1D};
use std::ops::Mul;

// impl<'a, 'b> Mul<&'b mut Tensor0D> for &'a mut Tensor0D {
//     type Output = Tensor0D;
//
//     fn mul(self, other: &'b mut Tensor0D) -> Self::Output {
//         println!("MUL");
//         let mut new = Tensor0D::new_without_tape(self.data * other.data);
//
//         new.tape = match (&self.tape, &other.tape) {
//             (Some(self_tape), Some(_other_tape)) => {
//                 let new_id = new.grad_for;
//                 let self_id = self.grad_for;
//                 let other_id = other.grad_for;
//                 let self_data = self.data;
//                 let other_data = other.data;
//                 self_tape.borrow_mut().add_operation((
//                     new_id,
//                     Box::new(move |g| {
//                         let mut tg1 = g.remove(new_id);
//                         let mut tg2 = tg1.clone();
//                         println!(
//                             "Mul Insert Before: {} {} {} {}",
//                             tg1.data, other_data, tg2.data, self_data
//                         );
//                         tg1.data *= other_data;
//                         tg2.data *= self_data;
//                         println!("Mul Insert: {} {}", tg1.data, tg2.data);
//                         g.insert(self_id, tg1);
//                         g.insert(other_id, tg2);
//                     }),
//                 ));
//                 Some(self_tape.clone())
//             }
//             (Some(self_tape), None) => {
//                 let new_id = new.grad_for;
//                 let self_id = self.grad_for;
//                 let other_data = other.data;
//                 self_tape.borrow_mut().add_operation((
//                     new_id,
//                     Box::new(move |g| {
//                         let mut tg = g.remove(new_id);
//                         tg.data *= other_data;
//                         println!("Mul Insert: {}", tg.data);
//                         g.insert(self_id, tg);
//                     }),
//                 ));
//                 Some(self_tape.clone())
//             }
//             (None, Some(other_tape)) => {
//                 // let new_id = new.grad_for;
//                 // let other_id = other.grad_for;
//                 // let self_data = self.data;
//                 // other_tape.borrow_mut().add_operation((
//                 //     new_id,
//                 //     Box::new(move |g| {
//                 //         let mut tg = g.remove(new_id);
//                 //         tg.data *= self_data;
//                 //         g.insert(other_id, tg);
//                 //     }),
//                 // ));
//                 // Some(other_tape.clone())
//                 panic!("WE HAVE NOT MADE THIS YET");
//             }
//             (None, None) => None,
//         };
//
//         new
//     }
// }

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
            (None, Some(other_tape)) => {
                // let new_id = new.grad_for;
                // let other_id = other.grad_for;
                // let self_data = self.data;
                // other_tape.borrow_mut().add_operation((
                //     new_id,
                //     Box::new(move |g| {
                //         let mut tg = g.remove(new_id);
                //         tg.data *= self_data;
                //         g.insert(other_id, tg);
                //     }),
                // ));
                // Some(other_tape.clone())
                panic!("WE HAVE NOT MADE THIS YET");
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

    const BATCH_SIZE: usize = 3;

    #[test]
    fn test_mul_1d() {
        let tape: Rc<RefCell<Tape<BATCH_SIZE>>> = Rc::new(RefCell::new(Tape::new()));
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
        let tape: Rc<RefCell<Tape<BATCH_SIZE>>> = Rc::new(RefCell::new(Tape::new()));
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

    // #[test]
    // fn test_mul_0d() {
    //     let tape = Rc::new(RefCell::new(Tape::new()));
    //     let mut a = Tensor0D::new_with_tape(1., Some(tape.clone()));
    //     let mut b = Tensor0D::new_with_tape(2., Some(tape.clone()));
    //     let mut c = &mut a * &mut b;
    //     // Check value match
    //     assert_eq!(2., c.data);
    //     // Check gradients
    //     let mut grads = c.backward();
    //     let a_grads = grads.remove(a.grad_for);
    //     let b_grads = grads.remove(b.grad_for);
    //     assert_eq!(2., a_grads.data);
    //     assert_eq!(1., b_grads.data);
    // }

    // #[test]
    // fn test_mul_1d_single_grad() {
    //     let tape = Rc::new(RefCell::new(Tape::new()));
    //     let mut a = Tensor0D::new_with_tape(2., Some(tape.clone()));
    //     let mut b = Tensor1D::new_without_tape([1., 2., 3.]);
    //     let mut c = &mut a * &mut b;
    //     // Check value match
    //     assert_eq!([2., 4., 6.], c.data);
    //     // Check gradients
    //     let mut grads = c.backward();
    //     let a_grads = grads.remove(a.grad_for);
    //     assert_eq!(6., a_grads.data);
    // }
    //
    // #[test]
    // fn test_mul_1d_dual_grad() {
    //     let tape = Rc::new(RefCell::new(Tape::new()));
    //     let mut a = Tensor0D::new_with_tape(2., Some(tape.clone()));
    //     let mut b = Tensor1D::new_without_tape([1., 2., 3.]);
    //     let mut c = Tensor0D::new_with_tape(3., Some(tape.clone()));
    //     let mut d = &mut a * &mut b;
    //     let mut e = &mut c * &mut d;
    //     // Check value match
    //     assert_eq!([6., 12., 18.], e.data);
    //     // Check gradients
    //     let mut grads = e.backward();
    //     let a_grads = grads.remove(a.grad_for);
    //     let c_grads = grads.remove(c.grad_for);
    //     assert_eq!(18., a_grads.data);
    //     assert_eq!(12., c_grads.data);
    // }
}
