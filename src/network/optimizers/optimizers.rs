// The most massive shoutout possible to Johnny's answer here: https://stackoverflow.com/questions/50017987/cant-clone-vecboxtrait-because-trait-cannot-be-made-into-an-object

pub trait Optimizer: OptimizerClone + Send {
    fn update(&mut self, g: f64) -> f64;
}

pub trait OptimizerClone {
    fn clone_box(&self) -> Box<dyn Optimizer>;
}

impl<T: 'static + Optimizer + Clone + Send> OptimizerClone for T {
    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Optimizer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
