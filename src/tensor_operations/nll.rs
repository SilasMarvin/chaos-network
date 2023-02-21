use crate::tensors::Tensor0D;

impl Tensor0D {
    pub fn nll(t: Vec<Self>, index: usize) -> Self {
        let sum_e = t.iter().fold(0., |acc, t| acc + t.data.exp());
        let log_softmax: Vec<f64> = t.iter().map(|t| (t.data.exp() / sum_e).ln()).collect();
        let loss = -1. * log_softmax[index];
        let mut new = Tensor0D::new_without_tape(loss);
        for (i, mut tensor) in t.into_iter().enumerate() {
            if let Some(tape) = tensor.tape.take() {
                let new_id = new.id;
                let self_id = tensor.id;
                let softmax_value = tensor.data.exp() / sum_e;
                let sub_one = index == i;
                tape.borrow_mut().add_operation((
                    new_id,
                    Box::new(move |g| {
                        let mut tg = g.remove(new_id);
                        if sub_one {
                            tg.data *= softmax_value - 1.;
                        } else {
                            tg.data *= softmax_value;
                        }
                        g.insert(self_id, tg);
                    }),
                ));
                if matches!(new.tape, None) {
                    new.tape = Some(tape);
                }
            }
        }
        new
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::gradients::Tape;
//     use crate::tensors::Tensor;
//     use std::cell::RefCell;
//     use std::rc::Rc;
//
//     #[test]
//     fn test_nll_0d() {
//         let tape = Some(Rc::new(RefCell::new(Tape::new())));
//         let a = vec![
//             Tensor0D::new_with_tape(1., tape.clone()),
//             Tensor0D::new_with_tape(2., tape.clone()),
//             Tensor0D::new_with_tape(3., tape.clone()),
//         ];
//         let ids = [a[0].id, a[1].id, a[2].id];
//         let mut b = Tensor0D::nll(a, 0);
//         // Check value match
//         assert_eq!(2.40760596444438, b.data);
//         // Check gradients
//         let mut grads = b.backward();
//         let a_0_grads = grads.remove(ids[0]);
//         let a_1_grads = grads.remove(ids[1]);
//         let a_2_grads = grads.remove(ids[2]);
//         assert_eq!(-0.9099694268296196, a_0_grads.data);
//         assert_eq!(0.24472847105479767, a_1_grads.data);
//         assert_eq!(0.6652409557748219, a_2_grads.data);
//     }
//
//     #[test]
//     fn test_nll_0d_more() {
//         let tape = Some(Rc::new(RefCell::new(Tape::new())));
//         let a = vec![
//             Tensor0D::new_with_tape(1., tape.clone()),
//             Tensor0D::new_with_tape(2., tape.clone()),
//             Tensor0D::new_with_tape(3., tape.clone()),
//         ];
//         let ids = [a[0].id, a[1].id, a[2].id];
//         let b = vec![
//             Tensor0D::new_without_tape(0.1),
//             Tensor0D::new_without_tape(0.2),
//             Tensor0D::new_without_tape(0.3),
//         ];
//         let c: Vec<Tensor0D> = a
//             .into_iter()
//             .zip(b.into_iter())
//             .map(|(mut a, mut b)| &mut a * &mut b)
//             .collect();
//         let mut d = Tensor0D::nll(c, 0);
//         // Check value match
//         assert_eq!(1.5206940689146358, d.data);
//         // Check gradients
//         let mut grads = d.backward();
//         let a_0_grads = grads.remove(ids[0]);
//         let a_1_grads = grads.remove(ids[1]);
//         let a_2_grads = grads.remove(ids[2]);
//         assert_eq!(-0.0781439861501746, a_0_grads.data);
//         assert_eq!(0.05900506558737987, a_1_grads.data);
//         assert_eq!(0.145924360069454, a_2_grads.data);
//     }
//
//     #[test]
//     fn test_nll_0d_even_more() {
//         let tape = Some(Rc::new(RefCell::new(Tape::new())));
//         let a = Tensor0D::new_with_tape(2.3, tape.clone());
//         let id = a.id;
//         let a = Tensor0D::split_on_add(a, 3);
//         let b = vec![
//             Tensor0D::new_without_tape(0.1),
//             Tensor0D::new_without_tape(0.2),
//             Tensor0D::new_without_tape(0.3),
//         ];
//         let c: Vec<Tensor0D> = a
//             .into_iter()
//             .zip(b.into_iter())
//             .map(|(mut a, mut b)| &mut a * &mut b)
//             .collect();
//         let mut d = Tensor0D::nll(c, 0);
//         // Check value match
//         assert_eq!(1.3461684771032714, d.data);
//         // Check gradients
//         let mut grads = d.backward();
//         let a_0_grads = grads.remove(id);
//         assert_eq!(0.11519967568849251, a_0_grads.data);
//     }
//
//     #[test]
//     fn test_nll_0d_even_even_more() {
//         let tape = Some(Rc::new(RefCell::new(Tape::new())));
//         let a = Tensor0D::new_with_tape(2.3, tape.clone());
//         let id = a.id;
//         let a = Tensor0D::split_on_add(a, 3);
//         let b = vec![
//             Tensor0D::new_without_tape(0.1),
//             Tensor0D::new_without_tape(0.2),
//             Tensor0D::new_without_tape(0.3),
//         ];
//         let c: Vec<Tensor0D> = a
//             .into_iter()
//             .zip(b.into_iter())
//             .map(|(mut a, mut b)| Tensor0D::mish(&mut (&mut a * &mut b)))
//             .collect();
//         let mut d = Tensor0D::nll(c, 0);
//         // Check value match
//         assert_eq!(1.30591406303897, d.data);
//         // Check gradients
//         let mut grads = d.backward();
//         let a_0_grads = grads.remove(id);
//         assert_eq!(0.11911438936456989, a_0_grads.data);
//     }
// }
