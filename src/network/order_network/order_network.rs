use crate::network::chaos_network::chaos_network::Node;
use crate::network::optimizers::AdamOptimizer;

pub trait OrderNetworkTrait<const I: usize, const O: usize, const N: usize>:
    OrderNetworkTraitClone<I, O, N> + Send + Sync
{
    fn forward_batch(&mut self, input: Box<[[f64; I]; N]>) -> Box<[[f64; O]; N]>;
    fn forward_batch_no_grad(&self, input: Box<[[f64; I]; N]>) -> Box<[[f64; O]; N]>;
    fn backwards(&mut self, grads: &[[f64; O]; N]) -> Vec<Vec<f64>>;
    fn apply_gradients(&mut self, grads: Vec<Vec<f64>>);
    fn merge_chaos_write_to_dir(&self, path: &str, chaos_network_nodes: Vec<Node<N>>);
}

pub trait OrderNetworkTraitClone<const I: usize, const O: usize, const N: usize> {
    fn clone_box(&self) -> Box<dyn OrderNetworkTrait<I, O, N>>;
}

impl<
        const I: usize,
        const O: usize,
        const N: usize,
        T: 'static + OrderNetworkTrait<I, O, N> + Clone + Send,
    > OrderNetworkTraitClone<I, O, N> for T
{
    fn clone_box(&self) -> Box<dyn OrderNetworkTrait<I, O, N>> {
        Box::new(self.clone())
    }
}

impl<const I: usize, const O: usize, const N: usize> Clone for Box<dyn OrderNetworkTrait<I, O, N>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub struct OrderNetwork<
    const D: usize,
    const W: usize,
    const I: usize,
    const O: usize,
    const N: usize,
> {
    pub weights: Vec<f64>,
    pub backwards: Option<Vec<Box<dyn FnOnce(&Vec<f64>, &[f64; O]) -> Vec<f64> + Send + Sync>>>,
    pub optimizer: AdamOptimizer<1, W>,
}

impl<const D: usize, const W: usize, const I: usize, const O: usize, const N: usize> Clone
    for OrderNetwork<D, W, I, O, N>
{
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            backwards: None,
            optimizer: self.optimizer.clone(),
        }
    }
}

#[macro_export]
macro_rules! build_order_network {
    ($f:literal, $i:literal, $o:literal, $n:literal) => {
        use std::fs::read_to_string;
        use std::fs::File;
        use std::io::Write;

        use crate::network::order_network::{OrderNetwork, OrderNetworkTrait};
        use crate::network::optimizers::AdamOptimizer;
        use crate::network::chaos_network::chaos_network::{Node, NodeKind};
        use chaos_network_derive::{build_backwards, build_forward, build_weights, get_weights_count};

        const WEIGHTS_COUNT: usize = get_weights_count!($f);

        impl Default for OrderNetwork<$f, WEIGHTS_COUNT, $i, $o, $n> {
            fn default() -> Self {
                Self {
                    weights: build_weights!($f),
                    backwards: None,
                    optimizer: AdamOptimizer::default()
                }
            }
        }

        impl OrderNetworkTrait<$i, $o, $n> for OrderNetwork<$f, WEIGHTS_COUNT, $i, $o, $n> {
            fn forward_batch_no_grad(&self, input: Box<[[f64; $i]; $n]>) -> Box<[[f64; $o]; $n]> {
                let output: Vec<[f64; $o]> = input.iter().map(|data| {
                    // let mut ret = Vec::new();
                    // ret.resize($o, 0.);
                    let mut ret = [0.; $o];
                    let weights = &self.weights;
                    build_forward!($f, weights, data, ret);
                    ret
                 }).collect();
                // unsafe {
                //     let output: Box<[[f64; $o]; $n]> = Box::from_raw(Box::into_raw(output.into_boxed_slice()) as * mut [[f64; $o]; $n]);
                //     output
                // }
                Box::new(output.try_into().unwrap())
            }

            fn forward_batch(&mut self, input: Box<[[f64; $i]; $n]>) -> Box<[[f64; $o]; $n]> {
                let mut output: Vec<[f64; $o]> = Vec::new();
                let mut partial_weight_funcs: Vec<
                    Box<dyn FnOnce(&Vec<f64>, &[f64; $o]) -> Vec<f64> + Send + Sync>,
                > = Vec::new();
                input.iter().enumerate().for_each(|(i, data)| {
                    let mut ret = [0.; $o];
                    let weights = &self.weights;
                    build_forward!($f, weights, data, ret);
                    output.push(ret);
                    let weights_len = self.weights.len();
                    let inputs = Box::new(data.to_owned());
                    partial_weight_funcs.push(Box::new(move |weights, output_grads| {
                        let mut weight_grads = Vec::new();
                        weight_grads.resize(weights_len, 0.);
                        build_backwards!($f, weights, inputs, output_grads, weight_grads);
                        weight_grads
                    }));
                });
                self.backwards = Some(partial_weight_funcs);
                Box::new(output.try_into().unwrap())
            }

            fn backwards(&mut self, grads: &[[f64; $o]; $n]) -> Vec<Vec<f64>> {
                match self.backwards.take() {
                    Some(vec) => {
                        let weights = &self.weights;
                        vec.into_iter()
                            .enumerate()
                            .map(|(i, func)| func(&weights, &grads[i]))
                            .collect()
                    }
                    None => panic!("Calling backwards on OrderNetwork when it has no grads"),
                }
            }

            fn apply_gradients(&mut self, batch_grads: Vec<Vec<f64>>) {
                let mut grads = [[0.; WEIGHTS_COUNT]; 1];
                for i in 0..$n {
                    for ii in 0..WEIGHTS_COUNT {
                        grads[0][ii] += batch_grads[i][ii] / ($n as f64);
                    }
                }
                let grads = self.optimizer.update(&grads);
                self.weights.iter_mut().zip(grads[0].into_iter()).for_each(|(w, g)| {
                    *w -= g;
                });
            }

            fn merge_chaos_write_to_dir(&self, path: &str, mut chaos_network_nodes: Vec<Node<$n>>) {
                let network_json = match read_to_string(format!("networks/{}/chaos-network.json", $f)) {
                    Ok(val) => val,
                    Err(_e) => {
                        panic!("Error merging and writing chaos network to dir")
                    }
                };
                let mut order_network_nodes: Vec<(NodeKind, Vec<usize>, Vec<f64>)> = match serde_json::from_str(&network_json) {
                    Ok(val) => val,
                    Err(_e) => panic!("Error merging and writing chaos network to dir")
                };
                // NOTE When merging we are removing the input nodes for the chaos network, and
                // changing the leaves of the order network to normal nodes with the connections
                // the chaos network input nodes had. This is why the input nodes do not have
                // biases in the chaos network
                // We also need to account for offset in the edges of the chaos network
                let edge_offset = order_network_nodes.len();
                let mut weights_taken = 0;
                order_network_nodes.iter_mut().for_each(|(_, _, w)| {
                    *w = self.weights[weights_taken..weights_taken + w.len()].to_owned();
                    weights_taken += w.len();
                });
                order_network_nodes.iter_mut().skip_while(|n| n.0 != NodeKind::Leaf).zip(chaos_network_nodes.iter()).for_each(|(on, cn)| {
                    on.0 = NodeKind::Normal;
                    on.1 = cn.edges.iter().map(|e| e + edge_offset - $o).collect::<Vec<usize>>();
                    on.2.append(&mut cn.weights.iter().map(|t| t.data).collect::<Vec<f64>>());
                });
                let mut chaos_network_nodes_to_keep: Vec<Node<$n>> = chaos_network_nodes.into_iter().skip_while(|n| n.kind == NodeKind::Input).collect();
                chaos_network_nodes_to_keep.iter_mut().for_each(|n| n.edges.iter_mut().for_each(|e| *e += edge_offset - $o));
                let mut chaos_network_nodes_to_keep: Vec<(NodeKind, Vec<usize>, Vec<f64>)> = chaos_network_nodes_to_keep.into_iter().map(|n|
                    (n.kind,
                    n.edges.clone(),
                    n.weights.iter().map(|w| w.data).collect::<Vec<f64>>())
                ).collect();
                order_network_nodes.append(&mut chaos_network_nodes_to_keep);
                // TODO tmr start by filtering out the unconnected normal nodes after the above
                // actions
                let write_out = serde_json::to_string(&order_network_nodes).unwrap();
                // Write it all out
                let path: String = format!("{}/chaos-network.json", path);
                let mut file = File::create(path).unwrap();
                write!(file, "{}", write_out).unwrap();

            }
        }
    };
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn prep_forward_batch_test_1() {
        let network_json = json!([
            ["Input", [2, 3], [0.1, 0.2]],
            ["Input", [2, 3], [0.3, 0.4]],
            ["Leaf", [], [0.5]],
            ["Leaf", [], [0.6]]
        ]);
        let mut file = File::create("networks/0/chaos-network.json").unwrap();
        write!(file, "{}", network_json).unwrap();
    }

    #[test]
    // NOTE compare output to results of test_forward_batch_1 in python-tests tests/order_network.py
    fn forward_batch_test_1() {
        build_order_network!(0, 2, 2, 2);
        const WEIGHTS_COUNT1: usize = get_weights_count!(0);
        let mut network: OrderNetwork<0, WEIGHTS_COUNT1, 2, 2, 2> = OrderNetwork::default();
        let input = Box::new([[0.1, 0.2], [0.2, 0.3]]);
        let output = network.forward_batch(input);
        assert_eq!(output[0], [0.4384104584592529, 0.5611483776438518]); // The rounding is messed up but this works
        let output_grads = [[0.13, 0.14], [0.14, 0.15]];
        let grads = network.backwards(&output_grads);
        assert_eq!(
            grads[0],
            vec![
                0.01193199182293912,
                0.013563128154802607,
                0.02386398364587824,
                0.027126256309605214,
                0.1193199182293912,
                0.13563128154802606
            ]
        );
        assert_eq!(
            grads[1],
            vec![
                0.02616781305329546,
                0.02966924163153822,
                0.03925171957994319,
                0.04450386244730733,
                0.1308390652664773,
                0.1483462081576911
            ]
        );
    }

    #[test]
    fn prep_forward_batch_test_2() {
        let network_json = json!([
            ["Input", [2, 3, 4], [0.1, 0.2, 0.3]],
            ["Input", [2, 3, 5], [0.4, 0.5, 0.6]],
            ["Normal", [4, 5], [0.7, 0.8, 0.9]],
            ["Normal", [4, 5], [0.1, 0.2, 0.3]],
            ["Leaf", [], [0.4]],
            ["Leaf", [], [0.5]]
        ]);
        let mut file = File::create("networks/1/chaos-network.json").unwrap();
        write!(file, "{}", network_json).unwrap();
    }

    #[test]
    // NOTE compare output to results of test_forward_batch_2 in python-tests tests/order_network.py
    fn forward_batch_test_2() {
        build_order_network!(1, 2, 2, 1);
        const WEIGHTS_COUNT2: usize = get_weights_count!(1);
        let mut network: OrderNetwork<1, WEIGHTS_COUNT2, 2, 2, 1> = OrderNetwork::default();
        let input = Box::new([[0.1, 0.2]]);
        let output = network.forward_batch(input);
        assert_eq!(output[0], [0.8433231723714711, 1.130676581160498]); // The rounding is messed up but this works
        let output_grads = [[0.13, 0.14]];
        let grads = network.backwards(&output_grads);
        assert_eq!(
            grads[0],
            vec![
                0.024439393766277528,
                0.005343493033151527,
                0.013586083416021974,
                0.048878787532555056,
                0.010686986066303054,
                0.030250713786655427,
                0.24439393766277526,
                0.08826906549591601,
                0.09826975717606541,
                0.053434930331515265,
                0.020000194963494835,
                0.022266173222685117,
                0.13586083416021974,
                0.15125356893327713,
            ]
        );
    }

    #[test]
    fn prep_forward_batch_test_3() {
        let network_json = json!([
            ["Input", [2, 3, 5], [0.1, 0.2, 0.3]],
            ["Input", [2, 3, 6], [0.4, 0.5, 0.6]],
            ["Normal", [5, 6, 4], [0.7, 0.8, 0.9, 0.1]],
            ["Normal", [5, 6, 4], [0.2, 0.3, 0.4, 0.5]],
            ["Normal", [5, 6], [0.6, 0.7, 0.8]],
            ["Leaf", [], [0.9]],
            ["Leaf", [], [0.1]]
        ]);
        let mut file = File::create("networks/2/chaos-network.json").unwrap();
        write!(file, "{}", network_json).unwrap();
    }

    #[test]
    // NOTE compare output to results of test_forward_batch_3 in python-tests tests/order_network.py
    fn forward_batch_test_3() {
        build_order_network!(2, 2, 2, 1);
        const WEIGHTS_COUNT3: usize = get_weights_count!(2);
        let mut network: OrderNetwork<2, WEIGHTS_COUNT3, 2, 2, 1> = OrderNetwork::default();
        let input = Box::new([[0.1, 0.2]]);
        let output = network.forward_batch(input);
        assert_eq!(output[0], [1.9038374515943743, 1.2983920430328038]); // The rounding is messed up but this works
        let output_grads = [[0.13, 0.14]];
        let grads = network.backwards(&output_grads);
        assert_eq!(
            grads[0],
            vec![
                0.02698030983733234,
                0.016810568409272402,
                0.013929013687361748,
                0.05396061967466468,
                0.033621136818544804,
                0.030451715204576336,
                0.2698030983733234,
                0.09049709057529431,
                0.09892271236484003,
                0.141668736099644,
                0.16810568409272403,
                0.031168961562968844,
                0.03407090990222471,
                0.048793473492833556,
                0.21805184582827822,
                0.08867297231388396,
                0.09692876178647514,
                0.13929013687361746,
                0.15225857602288168
            ]
        );
    }
}
