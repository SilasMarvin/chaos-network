use crate::tensors::{Tensor0D, Tensor1D};

impl Tensor0D {
    pub fn split_on_add(self, count: usize) -> Vec<Self> {
        match &self.tape {
            Some(tape) => {
                let mut new_tensors: Vec<Tensor0D> = (0..count)
                    .map(|_i| Tensor0D::new_with_tape(self.data, Some(tape.clone())))
                    .collect();
                for t in new_tensors.iter_mut() {
                    let self_id = t.grad_for;
                    let old_self_id = self.grad_for;
                    // For each split elment grab their gradients, and the parent gradients, and
                    // add them together
                    t.tape.as_mut().unwrap().borrow_mut().add_operation((
                        self_id,
                        Box::new(move |g| {
                            let tg = g.remove(self_id);
                            let mut tg2 = g.remove_or_0(old_self_id);
                            tg2.data += tg.data;
                            g.insert(old_self_id, tg2);
                        }),
                    ));
                }
                new_tensors
            }
            None => (0..count)
                .map(|_i| Tensor0D::new_without_tape(self.data))
                .collect(),
        }
    }
}

impl<const N: usize> Tensor1D<N> {
    pub fn split_on_add(self, count: usize) -> Vec<Self> {
        match &self.tape {
            Some(tape) => {
                let mut new_tensors: Vec<Tensor1D<N>> = (0..count)
                    .map(|_i| Self::new_with_tape(self.data, Some(tape.clone())))
                    .collect();
                for t in new_tensors.iter_mut() {
                    let self_id = t.grad_for;
                    let old_self_id = self.grad_for;
                    // For each split elment grab their gradients, and the parent gradients, and
                    // add them together
                    t.tape.as_mut().unwrap().borrow_mut().add_operation((
                        self_id,
                        Box::new(move |g| {
                            let tg = g.remove(self_id);
                            let mut tg2 = g.remove_or_0(old_self_id);
                            tg2.data += tg.data;
                            g.insert(old_self_id, tg2);
                        }),
                    ));
                }
                new_tensors
            }
            None => (0..count)
                .map(|_i| Tensor1D::new_without_tape(self.data))
                .collect(),
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::tensors::Tensor;
//
//     #[test]
//     fn test_add_0d() {
//         let a = Tensor0D::new_with_tape(1.);
//         let a_id = a.id;
//         let mut b = Tensor0D::new_without_tape(2.);
//         let mut c = Tensor0D::new_without_tape(3.);
//         let mut d = a.split_on_add(2);
//         let mut e = &mut (&mut b * &mut d[0]) + &mut (&mut c * &mut d[1]);
//         // Check value match
//         assert_eq!(5., e.data);
//         // Check gradients
//         let mut grads = e.backward();
//         let a_grads = grads.remove(a_id);
//         assert_eq!(5., a_grads.data);
//     }
//
//     #[test]
//     fn test_add_0d_2() {
//         let a = Tensor0D::new_with_tape(2.);
//         let a_id = a.id;
//         let mut b = Tensor0D::new_without_tape(3.);
//         let mut c = Tensor0D::new_without_tape(9.);
//         let mut d = a.split_on_add(3);
//         let mut e = &mut (&mut (&mut b + &mut d[0]) * &mut d[1]) * &mut (&mut c * &mut d[2]);
//         // Check value match
//         assert_eq!(180., e.data);
//         // Check gradients
//         let mut grads = e.backward();
//         let a_grads = grads.remove(a_id);
//         assert_eq!(216., a_grads.data);
//     }
//
//     #[test]
//     fn test_add_0d_3() {
//         let a = Tensor0D::new_with_tape(2.);
//         let a_id = a.id;
//         let mut b = Tensor0D::new_without_tape(0.3);
//         let mut c = Tensor0D::new_without_tape(0.9);
//         let mut d = a.split_on_add(2);
//         let mut e = &mut b * &mut d[0];
//         let mut f = &mut c * &mut d[1];
//         let mut l = Tensor0D::nll(vec![e, f], 0);
//         // Check value match
//         assert_eq!(1.4632823, l.data);
//         // Check gradients
//         let mut grads = l.backward();
//         let a_grads = grads.remove(a_id);
//         assert_eq!(0.46111482, a_grads.data);
//     }
// }
