use crate::tensors::Tensor0D;


impl Tensor0D {
    pub fn exp(&mut self) -> Self {
        Tensor0D::new_with_tape(self.data.exp())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_0d() {
        let mut a = Tensor0D::new_with_tape(1.);
        let b = a.exp();
        assert_eq!(b.data, 1f32.exp());
    }
}
