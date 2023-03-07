use crate::network::optimizers::Optimizer;

#[derive(Clone)]
pub struct AdamOptimizer {
    a: f64,
    b1: f64,
    b2: f64,
    e: f64,
    m: f64,
    v: f64,
    t: i32,
}

impl Default for AdamOptimizer {
    fn default() -> Self {
        Self {
            a: 0.001,
            b1: 0.9,
            b2: 0.999,
            e: 1e-8,
            m: 0.,
            v: 0.,
            t: 0,
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn update(&mut self, g: f64) -> f64 {
        self.t += 1;
        self.m = self.b1 * self.m + (1. - self.b1) * g;
        self.v = self.b2 * self.v + (1. - self.b2) * g.powi(2);
        let m_hat = self.m / (1. - self.b1.powi(self.t));
        let v_hat = self.v / (1. - self.b2.powi(self.t));
        (self.a * m_hat) / (v_hat.sqrt() + self.e)
    }
}
