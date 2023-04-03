use crate::network::chaos_network::gradients::Tape;
use crate::network::chaos_network::tensors::{element_wise_addition, Tensor1D, WithTape};

pub trait Tensor1DSplitOnAdd<const N: usize, TensorTape1> {
    fn split_on_add(&self, count: usize, tape: &mut Tape<N>) -> Vec<Tensor1D<N, TensorTape1>>;
}

impl<const N: usize> Tensor1DSplitOnAdd<N, WithTape> for Tensor1D<N, WithTape> {
    fn split_on_add(&self, count: usize, tape: &mut Tape<N>) -> Vec<Tensor1D<N, WithTape>> {
        let mut new_tensors: Vec<Tensor1D<N, WithTape>> =
            (0..count).map(|_i| Self::new(self.data)).collect();

        // Add operation to tape
        // For each split elment grab their gradients, and the parent gradients, and add them together
        for t in new_tensors.iter_mut() {
            t.set_id_grad_for(tape.get_next_temporary_tensor_id());
            let self_id = t.grad_for;
            let old_self_id = self.grad_for;
            tape.add_operation((
                self_id,
                Box::new(move |g| {
                    let tg = g.remove(self_id);
                    let mut tg2 = g.remove_or_0(old_self_id);
                    tg2.data = *element_wise_addition::<N>(&tg2.data, &tg.data);
                    g.insert(old_self_id, tg2);
                }),
            ));
        }

        new_tensors
    }
}
