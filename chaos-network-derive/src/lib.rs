use proc_macro;
use proc_macro2::TokenStream;

// use chaos_network::NodeKind;

use quote::{format_ident, quote, quote_spanned};
use serde::Deserialize;
use syn::{parse_macro_input, DeriveInput};

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Deserialize, PartialEq, Eq, Clone, Copy)]
enum NodeKind {
    Input,
    Normal,
    Leaf,
}

#[proc_macro]
pub fn build_weights(input_token_stream: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let dn: Vec<(NodeKind, Vec<usize>, Vec<f64>)> =
        serde_json::from_str(&include_str!("../../layers/1234/layer1.txt")).unwrap();
    let mut ret: Vec<f64> = Vec::new();
    for ((_, _, mut weights)) in dn.into_iter() {
        ret.append(&mut weights);
    }
    proc_macro::TokenStream::from(quote! {
       vec![#(#ret),*]
    })
}

#[proc_macro]
pub fn build_forward(input_token_stream: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = proc_macro2::TokenStream::from(input_token_stream);
    let mut input = input.into_iter();
    let weights_ident = input.next().unwrap();
    let _ = input.next().unwrap();
    let input_ident = input.next().unwrap();
    let _ = input.next().unwrap();
    let ret_ident = input.next().unwrap();

    let dn: Vec<(NodeKind, Vec<usize>, Vec<f64>)> =
        serde_json::from_str(&include_str!("../../layers/1234/layer1.txt")).unwrap();

    let mut ke: HashMap<usize, Vec<(usize, NodeKind, usize)>> = HashMap::new();
    let mut weight_index = 0;
    let mut ret = Vec::new();
    let mut leaves_count: usize = 0;
    for (i, (kind, edges, weights)) in dn.into_iter().enumerate() {
        if kind == NodeKind::Input {
            // Setup further edges
            edges
                .into_iter()
                .zip(weights.into_iter())
                .for_each(|(e, _)| {
                    match ke.get_mut(&e) {
                        Some(v) => v.push((i, kind, weight_index)),
                        None => drop(ke.insert(e, vec![(i, kind, weight_index)])),
                    }
                    weight_index += 1;
                });
        } else if kind == NodeKind::Normal {
            let adds =
                ke.remove(&i)
                    .unwrap()
                    .into_iter()
                    .map(|(node_index, kind, weight_index)| {
                        if kind == NodeKind::Input {
                            quote! {
                                input[#node_index] * #weights_ident.weights[#weight_index]
                            }
                        } else {
                            let varname = format_ident!("x{}", node_index);
                            quote! {
                                #varname * #weights_ident.weights[#weight_index]
                            }
                        }
                    });
            let varname = format_ident!("x{}", i);
            ret.push(quote! {
                let #varname = #(#adds)+*;
            });
            // Setup further edges
            edges
                .into_iter()
                .zip(weights.into_iter().skip(1))
                .for_each(|(e, _)| {
                    match ke.get_mut(&e) {
                        Some(v) => v.push((i, kind, weight_index)),
                        None => drop(ke.insert(e, vec![(i, kind, weight_index)])),
                    }
                    weight_index += 1;
                });
        } else {
            let adds = match ke.remove(&i) {
                Some(ee) => ee
                    .into_iter()
                    .map(|(node_index, kind, weight_index)| {
                        if kind == NodeKind::Input {
                            quote! {
                                #input_ident[#node_index] * #weights_ident.weights[#weight_index]
                            }
                        } else {
                            let varname = format_ident!("x{}", node_index);
                            quote! {
                                #varname * #weights_ident.weights[#weight_index]
                            }
                        }
                    })
                    .collect(),
                None => Vec::new(),
            };
            if !adds.is_empty() {
                ret.push(quote! {
                    #ret_ident[#leaves_count] = #(#adds)+*;
                });
            }
            leaves_count += 1;
        }
    }

    ret.push(quote! {
        let x = 5;
    });

    proc_macro::TokenStream::from(quote! {
        #(#ret)*
    })

    // "fn forward() -> u32 { 42 }".parse().unwrap()
}
