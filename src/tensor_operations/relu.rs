use crate::tensors::{element_wise_mul, Tensor, Tensor1D};

impl<const N: usize> Tensor1D<N> {
    pub fn relu(t: &mut Self) -> Self {
        let data = t.data.map(|x| if x > 0. { x } else { 0. });
        let mut new = Tensor1D::new_with_tape(data, t.tape.clone());

        if let Some(tape) = &t.tape {
            new.set_tape(Some(tape.clone()));
            let new_id = new.grad_for;
            let self_id = t.grad_for;
            let t_data = t.data.map(|x| if x > 0. { 1. } else { 0. });
            tape.write().add_operation((
                new_id,
                Box::new(move |g| {
                    let mut tg = g.remove(new_id);
                    tg.data = element_wise_mul::<N>(&tg.data, &t_data);
                    g.insert(self_id, tg);
                }),
            ));
        }

        new
    }
}
