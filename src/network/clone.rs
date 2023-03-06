use crate::gradients::Tape;
use crate::network::Network;
use std::sync::Arc;
use std::sync::RwLock;

use super::NetworkMode;

impl<const N: usize> Clone for Network<N> {
    fn clone(&self) -> Self {
        let mut x = Self {
            inputs_count: self.inputs_count,
            leaves_count: self.leaves_count,
            nodes: self.nodes.clone(),
            connections_to: self.connections_to.clone(),
            mode: self.mode,
            tape: Arc::new(RwLock::new(Tape::new())),
        };
        // This handles removing the reference to the previous networks tape from nodes
        x.set_mode(NetworkMode::Inference);
        x
    }
}
