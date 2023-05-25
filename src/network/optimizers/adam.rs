#[derive(Clone)]
pub struct AdamOptimizer<const I: usize, const O: usize> {
    a: f64,
    b1: f64,
    b2: f64,
    b1t: f64,
    b2t: f64,
    e: f64,
    m: [[f64; O]; I],
    v: [[f64; O]; I],
}

impl<const I: usize, const O: usize> Default for AdamOptimizer<I, O> {
    fn default() -> Self {
        Self {
            a: 0.00025,
            b1: 0.9,
            b2: 0.999,
            b1t: 0.9,
            b2t: 0.999,
            e: 10e-8,
            m: [[0.; O]; I],
            v: [[0.; O]; I],
        }
    }
}

impl<const I: usize, const O: usize> AdamOptimizer<I, O> {
    fn new(a: f64, b1: f64, b2: f64, e: f64) -> Self {
        Self {
            a,
            b1,
            b2,
            b1t: b1,
            b2t: b2,
            e,
            m: [[0.; O]; I],
            v: [[0.; O]; I],
        }
    }
}

impl<const I: usize, const O: usize> AdamOptimizer<I, O> {
    pub fn update(&mut self, g: &[[f64; O]; I]) -> Box<[[f64; O]; I]> {
        self.b1t = self.b1t * self.b1;
        self.b2t = self.b2t * self.b2;
        let mut ret = [[0.; O]; I];
        for i in 0..I {
            for ii in 0..O {
                self.m[i][ii] = self.b1 * self.m[i][ii] + (1. - self.b1) * g[i][ii];
                self.v[i][ii] = self.b2 * self.v[i][ii] + (1. - self.b2) * (g[i][ii] * g[i][ii]);
                let m_hat = self.m[i][ii] / (1. - self.b1t);
                let v_hat = self.v[i][ii] / (1. - self.b2t);
                ret[i][ii] = self.a * (m_hat / (v_hat.sqrt() + self.e))
            }
        }
        Box::new(ret)
    }
}
