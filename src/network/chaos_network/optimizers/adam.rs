use crate::network::chaos_network::optimizers::Optimizer;

#[derive(Clone)]
pub struct AdamOptimizer {
    a: f64,
    b1: f64,
    b2: f64,
    b1t: f64,
    b2t: f64,
    e: f64,
    m: f64,
    v: f64,
}

impl Default for AdamOptimizer {
    fn default() -> Self {
        Self {
            a: 0.00025,
            b1: 0.9,
            b2: 0.999,
            b1t: 0.9,
            b2t: 0.999,
            e: 10e-8,
            m: 0.,
            v: 0.,
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn update(&mut self, g: f64) -> f64 {
        self.m = self.b1 * self.m + (1. - self.b1) * g;
        self.v = self.b2 * self.v + (1. - self.b2) * (g * g);
        self.b1t = self.b1t * self.b1;
        self.b2t = self.b2t * self.b2;
        let m_hat = self.m / (1. - self.b1t);
        let v_hat = self.v / (1. - self.b2t);
        (self.a * m_hat) / (v_hat.sqrt() + self.e)
    }
}
