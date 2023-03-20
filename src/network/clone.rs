use crate::gradients::Tape;
use crate::network::Network;

impl<const N: usize> Clone for Network<N> {
    fn clone(&self) -> Self {
        Self {
            inputs_count: self.inputs_count,
            leaves_count: self.leaves_count,
            nodes: self.nodes.clone(),
            tape: self.tape.clone(),
            input_connectivity_chance: self.input_connectivity_chance,
        }
    }
}
