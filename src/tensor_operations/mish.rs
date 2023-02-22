use crate::tensors::{Tensor0D, Tensor1D};

fn do_mish_backward(x: f64) -> f64 {
    let w = (4. * (x + 1.)) + (4. * (2. * x).exp()) + (3. * x).exp() + (x.exp() * ((4. * x) + 6.));
    let d = (2. * x.exp()) + (2. * x).exp() + 2.;
    (x.exp() * w) / d.powi(2)
}

impl Tensor0D {
    pub fn mish(t: &mut Self) -> Self {
        let data = t.data * ((1. + t.data.exp()).ln()).tanh();
        let mut new = Tensor0D::new_without_tape(data);

        if let Some(tape) = &t.tape {
            let new_id = new.grad_for;
            let self_id = t.grad_for;
            let t_data = t.data;
            tape.borrow_mut().add_operation((
                new_id,
                Box::new(move |g| {
                    let mut tg = g.remove(new_id);
                    tg.data *= do_mish_backward(t_data);
                    println!("Mish insert: {}", tg.data);
                    g.insert(self_id, tg);
                }),
            ));
            new.tape = Some(tape.clone());
        }

        new
    }
}

impl<const N: usize> Tensor1D<N> {
    pub fn mish(t: &mut Self) -> Self {
        let data = t.data.map(|x| x * ((1. + x.exp()).ln()).tanh());
        let mut new = Tensor1D::new_with_tape(data, t.tape.clone());

        if let Some(tape) = &t.tape {
            let new_id = new.grad_for;
            let self_id = t.grad_for;
            let t_data: f64 = t.data.map(do_mish_backward).iter().sum();
            tape.borrow_mut().add_operation((
                new_id,
                Box::new(move |g| {
                    let mut tg = g.remove(new_id);
                    tg.data *= t_data;
                    println!("Mish insert: {}", tg.data);
                    g.insert(self_id, tg);
                }),
            ));
            new.tape = Some(tape.clone());
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
    fn test_mish_0d() {
        let tape = Rc::new(RefCell::new(Tape::new()));
        let mut a = Tensor0D::new_with_tape(1., Some(tape.clone()));
        let mut b = Tensor0D::mish(&mut a);
        // Check value match
        assert_eq!(0.8650983882673103, b.data);
        // Check gradients
        let mut grads = b.backward();
        let a_grads = grads.remove(a.id);
        assert_eq!(1.0490362200997918, a_grads.data);
    }

    // Might have to review this one later
    #[test]
    fn test_mish_1d() {
        let tape = Rc::new(RefCell::new(Tape::new()));
        let mut a = Tensor0D::new_with_tape(2., Some(tape.clone()));
        let mut b = Tensor1D::new_without_tape([1., 2., 3.]);
        let mut c = Tensor1D::mish(&mut (&mut a * &mut b));
        // Check value match
        assert_eq!(
            [1.9439589595339946, 3.9974128069762385, 5.999926634065252],
            c.data
        );
        // Check gradients
        let mut grads = c.backward();
        let a_grads = grads.remove(a.grad_for);
        assert_eq!(18.443309710136695, a_grads.data);
    }
}
