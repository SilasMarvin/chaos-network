#[inline(always)]
pub fn element_wise_mul<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    let mut new = [0.; N];
    for i in 0..N {
        new[i] = a[i] * b[i];
    }
    new
}

#[inline(always)]
pub fn element_wise_addition<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    let mut new = [0.; N];
    for i in 0..N {
        new[i] = a[i] + b[i];
    }
    new
}
