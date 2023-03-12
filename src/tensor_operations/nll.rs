use crate::gradients::Tape;
use crate::tensors::{element_wise_mul, Tensor1D, WithTape};

pub trait Tensor1DNll<const N: usize> {
    fn nll(
        t: Vec<Tensor1D<N, WithTape>>,
        indexes: &Vec<usize>,
        tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithTape>;
}

impl<const N: usize> Tensor1DNll<N> for Tensor1D<N, WithTape> {
    fn nll(
        t: Vec<Tensor1D<N, WithTape>>,
        indexes: &Vec<usize>,
        tape: &mut Tape<N>,
    ) -> Tensor1D<N, WithTape> {
        let sum_e = t.iter().fold([0.; N], |mut acc, t| {
            t.data
                .iter()
                .enumerate()
                .for_each(|(i, x)| acc[i] += x.exp());
            acc
        });
        let log_softmax: Vec<[f64; N]> = t
            .iter()
            .map(|t| {
                let mut tracker = 0;
                t.data.map(|x| {
                    let x = (x.exp() / sum_e[tracker]).ln();
                    tracker += 1;
                    x
                })
            })
            .collect();
        let losses: [f64; N] = indexes
            .iter()
            .enumerate()
            .map(|(ii, i)| -1. * log_softmax[*i][ii])
            .collect::<Vec<f64>>()
            .try_into()
            .unwrap();
        let mut new = Tensor1D::new(losses);
        new.set_id_grad_for(tape.get_next_temporary_tensor_id());
        for (i, tensor) in t.into_iter().enumerate() {
            let new_id = new.grad_for;
            let self_id = tensor.grad_for;
            let mut tracker = 0;
            let softmax_value = tensor.data.map(|x| {
                let mut x = x.exp() / sum_e[tracker];
                if i == indexes[tracker] {
                    x -= 1.;
                }
                tracker += 1;
                x
            });
            tape.add_operation((
                new_id,
                Box::new(move |g| {
                    let mut tg = g.remove(new_id);
                    tg.data = element_wise_mul::<N>(&tg.data, &softmax_value);
                    g.insert(self_id, tg);
                }),
            ));
        }
        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradients::Tape;

    #[test]
    // Might need to double check this one
    fn test_nll_1d() {
        let mut tape: Tape<3> = Tape::new();
        let mut a: Vec<Tensor1D<3, WithTape>> = vec![
            Tensor1D::new([1.; 3]),
            Tensor1D::new([2.; 3]),
            Tensor1D::new([3.; 3]),
        ];
        a.iter_mut()
            .for_each(|t| t.set_id_grad_for(tape.get_next_temporary_tensor_id()));
        let b = Tensor1D::nll(a, &vec![0, 1, 2], &mut tape);
        assert_eq!(2.40760596444438, b.data[0]);
        assert_eq!(1.4076059644443801, b.data[1]);
        assert_eq!(0.4076059644443803, b.data[2]);
        let grads = tape.execute();
        let a_0_grads = grads.remove(1);
        let a_1_grads = grads.remove(2);
        let a_2_grads = grads.remove(3);
        assert_eq!(
            [
                -0.9099694268296196,
                0.09003057317038046,
                0.09003057317038046
            ],
            a_0_grads.data
        );
        assert_eq!(
            [
                0.24472847105479767,
                -0.7552715289452023,
                0.24472847105479767
            ],
            a_1_grads.data
        );
        assert_eq!(
            [0.6652409557748219, 0.6652409557748219, -0.3347590442251781],
            a_2_grads.data
        );
    }
}
