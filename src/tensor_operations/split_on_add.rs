use crate::tensors::{element_wise_addition, Tensor1D};

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
                    t.tape.as_mut().unwrap().write().unwrap().add_operation((
                        self_id,
                        Box::new(move |g| {
                            let tg = g.remove(self_id);
                            let mut tg2 = g.remove_or_0(old_self_id);
                            tg2.data = element_wise_addition::<N>(&tg2.data, &tg.data);
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
