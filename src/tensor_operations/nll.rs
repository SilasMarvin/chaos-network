use crate::gradients::Tape;
use crate::tensors::Tensor0D;

impl Tensor0D {
    pub fn nll(t: Vec<Self>, index: usize) -> Self {
        let sum_e = t.iter().fold(0., |acc, t| acc + t.data.exp());
        let log_softmax: Vec<f32> = t.iter().map(|t| (t.data.exp() / sum_e).ln()).collect();
        let loss = -1. * log_softmax[index];
        let mut new = Tensor0D::new_without_tape(loss);
        let mut new_tape = Tape::new();
        for (i, mut tensor) in t.into_iter().enumerate() {
            if let Some(tape) = tensor.tape.take() {
                new_tape.merge(tape);
                let new_id = new.id;
                let self_id = tensor.id;
                let softmax_value = tensor.data.exp() / sum_e;
                let sub_one = index == i;
                new_tape.add_operation(Box::new(move |g| {
                    let mut tg = g.remove(new_id);
                    if sub_one {
                        tg.data *= softmax_value - 1.;
                    } else {
                        tg.data *= softmax_value;
                    }
                    g.insert(self_id, tg);
                }));
            }
        }
        if new_tape.operation_count() > 0 {
            new.tape = Some(new_tape);
        }
        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensors::Tensor;

    #[test]
    fn test_relu_0d() {
        let a = vec![
            Tensor0D::new_with_tape(1.),
            Tensor0D::new_with_tape(2.),
            Tensor0D::new_with_tape(3.),
        ];
        let ids = [a[0].id, a[1].id, a[2].id];
        let mut b = Tensor0D::nll(a, 0);
        // Check value match
        assert_eq!(2.407606, b.data);
        // Check gradients
        let mut grads = b.backward();
        let a_0_grads = grads.remove(ids[0]);
        let a_1_grads = grads.remove(ids[1]);
        let a_2_grads = grads.remove(ids[2]);
        assert_eq!(-0.90996945, a_0_grads.data);
        assert_eq!(0.24472848, a_1_grads.data);
        assert_eq!(0.66524094, a_2_grads.data);
    }
}
