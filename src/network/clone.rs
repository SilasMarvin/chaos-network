use crate::gradients::Tape;
use crate::network::Network;

impl<const N: usize> Clone for Network<N> {
    fn clone(&self) -> Self {
        Self {
            inputs_count: self.inputs_count,
            leaves_count: self.leaves_count,
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            edges_to_count: self.edges_to_count.clone(),
            tape: self.tape.clone(),
        }
    }
}
