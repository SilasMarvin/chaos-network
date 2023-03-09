use crate::network::{Network, Node};
use std::fmt;

impl<const N: usize> fmt::Debug for Node<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Node: {} kind: {:?}", self.id, self.kind)
    }
}

impl<const N: usize> fmt::Debug for Network<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "
Network:
    - input_nodes: {}
    - normal nodes: {}
    - leaves: {}
    - connections: {}
        ",
            self.inputs_count,
            self.nodes.len() - self.inputs_count - self.leaves_count,
            self.leaves_count,
            self.get_connection_count(),
        )
    }
}
