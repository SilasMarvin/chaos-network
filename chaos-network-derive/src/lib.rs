use proc_macro::TokenStream;

// use chaos_network::NodeKind;
use quote::{format_ident, quote, quote_spanned};
use serde::Deserialize;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Ident, LitInt};

use std::collections::HashMap;
use std::fs::read_to_string;

#[derive(Debug, Deserialize, PartialEq, Eq, Clone, Copy)]
enum NodeKind {
    Input,
    Normal,
    Leaf,
}

#[proc_macro]
pub fn build_weights(input_token_stream: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(input_token_stream as LitInt);
    let file_dir: i32 = match ast.base10_parse() {
        Ok(val) => val,
        Err(_e) => {
            return TokenStream::from(
                quote_spanned! { ast.span() =>  compile_error!("build_weights requires a LitInt") },
            )
        }
    };
    let network_json = match read_to_string(format!("networks/{}/1.json", file_dir)) {
        Ok(val) => val,
        Err(_e) => {
            return TokenStream::from(
                quote_spanned! { ast.span() =>  compile_error!("The requested networks file does not exist") },
            )
        }
    };
    let dn: Vec<(NodeKind, Vec<usize>, Vec<f64>)> = match serde_json::from_str(&network_json) {
        Ok(val) => val,
        Err(_e) => return TokenStream::from(quote! { Vec::new() }),
    };
    let mut ret: Vec<f64> = Vec::new();
    for (_, _, mut weights) in dn.into_iter() {
        ret.append(&mut weights);
    }
    TokenStream::from(quote! {
        vec![#(#ret),*]
    })
}

struct BuildForwardArgs {
    file_dir: i32,
    weights_ident: Ident,
    input_ident: Ident,
    ret_ident: Ident,
}

impl Parse for BuildForwardArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let file_dir: LitInt = input.parse()?;
        let file_dir = file_dir.base10_parse()?;
        input.parse::<syn::Token![,]>()?;
        let weights_ident = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let input_ident = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let ret_ident = input.parse()?;
        Ok(BuildForwardArgs {
            file_dir,
            weights_ident,
            input_ident,
            ret_ident,
        })
    }
}

#[proc_macro]
pub fn build_forward(input: TokenStream) -> TokenStream {
    let BuildForwardArgs {
        file_dir,
        weights_ident,
        input_ident,
        ret_ident,
    } = parse_macro_input!(input as BuildForwardArgs);

    let network_json = match read_to_string(format!("networks/{}/1.json", file_dir)) {
        Ok(val) => val,
        Err(_e) => {
            return TokenStream::from(
                quote_spanned! { weights_ident.span() =>  compile_error!("The requested networks file does not exist") },
            )
        }
    };
    let dn: Vec<(NodeKind, Vec<usize>, Vec<f64>)> = match serde_json::from_str(&network_json) {
        Ok(val) => val,
        Err(_e) => {
            return TokenStream::from(
                quote_spanned! { weights_ident.span() =>  compile_error!("The networks file is not valid json") },
            )
        }
    };

    let mut ke: HashMap<usize, Vec<(usize, NodeKind, usize)>> = HashMap::new();
    let mut weight_index = 0;
    let mut ret = Vec::new();
    let mut leaves_count: usize = 0;
    for (i, (kind, edges, _)) in dn.into_iter().enumerate() {
        if kind == NodeKind::Input {
            // Setup further edges
            edges.into_iter().for_each(|e| {
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
                                #input_ident[#node_index] * #weights_ident[#weight_index]
                            }
                        } else {
                            let varname = format_ident!("x{}", node_index);

                            quote! {
                                #varname * #weights_ident[#weight_index]
                            }
                        }
                    });
            // Make sure we account for the bias
            let varname = format_ident!("x{}", i);
            ret.push(quote! {
                let #varname = #(#adds)+* + #weights_ident[weight_index];
            });
            weight_index += 1;
            // Setup further edges
            edges.into_iter().for_each(|e| {
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
                                #input_ident[#node_index] * #weights_ident[#weight_index]
                            }
                        } else {
                            let varname = format_ident!("x{}", node_index);
                            quote! {
                                #varname * #weights_ident[#weight_index]
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

    TokenStream::from(quote! {
        #(#ret)*
    })

    // "fn forward() -> u32 { 42 }".parse().unwrap()
}

struct BuildBackwardsArgs {
    file_dir: i32,
    weights_ident: Ident,
    input_ident: Ident,
    output_grads_ident: Ident,
    weight_grads_ident: Ident,
}

impl Parse for BuildBackwardsArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let file_dir: LitInt = input.parse()?;
        let file_dir = file_dir.base10_parse()?;
        input.parse::<syn::Token![,]>()?;
        let weights_ident = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let input_ident = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let output_grads_ident = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let weight_grads_ident = input.parse()?;
        Ok(BuildBackwardsArgs {
            file_dir,
            weights_ident,
            input_ident,
            output_grads_ident,
            weight_grads_ident,
        })
    }
}

#[proc_macro]
pub fn build_backwards(input_token_stream: TokenStream) -> TokenStream {
    let BuildBackwardsArgs {
        file_dir,
        weights_ident,
        input_ident,
        output_grads_ident,
        weight_grads_ident,
    } = parse_macro_input!(input_token_stream as BuildBackwardsArgs);

    let network_json = match read_to_string(format!("networks/{}/1.json", file_dir)) {
        Ok(val) => val,
        Err(_e) => {
            return TokenStream::from(
                quote_spanned! { weights_ident.span() =>  compile_error!("The requested networks file does not exist") },
            )
        }
    };
    let dn: Vec<(NodeKind, Vec<usize>, Vec<f64>)> = match serde_json::from_str(&network_json) {
        Ok(val) => val,
        Err(_e) => {
            return TokenStream::from(
                quote_spanned! { weights_ident.span() =>  compile_error!("The networks file is not valid json") },
            )
        }
    };

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
            let dx_varname = format_ident!("dx{}", i);
            let adds = ke.remove(&i).unwrap().into_iter().map(
                |(node_index, kind, weight_index)| {
                    if kind == NodeKind::Input {
                        quote! {
                            #weight_grads_ident[#weight_index] = #input_ident[#node_index] * #dx_varname;
                        }
                    } else {
                        let varname = format_ident!("x{}", node_index);
                        quote! {
                            #weight_grads_ident[#weight_index] = #varname * #dx_varname;
                        }
                    }
                },
            );
            // Include the bias, it is manually inserted below
            let bias_index = weight_index;
            weight_index += 1;
            // Insert more edges and build the dx for the current index
            let mut dx_varbuilder = Vec::new();
            edges.into_iter().for_each(|e| {
                match ke.get_mut(&e) {
                    Some(v) => v.push((i, kind, weight_index)),
                    None => drop(ke.insert(e, vec![(i, kind, weight_index)])),
                }
                let inner_dx_varname = format_ident!("dx{}", e);
                dx_varbuilder.push(quote! {
                    #weights_ident[#weight_index] * #inner_dx_varname
                });
                weight_index += 1;
            });
            // Insert this stuff and continue
            ret.insert(
                0,
                quote! {
                    let #dx_varname = #(#dx_varbuilder)+*;
                    #weight_grads_ident[#bias_index] = #dx_varname; // The bias
                    #(#adds)*
                },
            );
        } else {
            let dx_varname = format_ident!("dx{}", i);
            let adds = match ke.remove(&i) {
                Some(ee) => ee
                    .into_iter()
                    .map(|(node_index, kind, weight_index)| {
                        if kind == NodeKind::Input {
                            quote! {
                                #weight_grads_ident[#weight_index] = #input_ident[#node_index] * #dx_varname;
                            }
                        } else {
                            let varname = format_ident!("x{}", node_index);
                            quote! {
                                #weight_grads_ident[#weight_index] = #varname * #dx_varname;
                            }
                        }
                    })
                    .collect(),
                None => Vec::new(),
            };
            if !adds.is_empty() {
                ret.insert(
                    0,
                    quote! {
                        let #dx_varname = #output_grads_ident[#leaves_count];
                        #(#adds)*
                    },
                );
            }
            leaves_count += 1;
        }
    }

    TokenStream::from(quote! {
        #(#ret)*
    })

    // "fn forward() -> u32 { 42 }".parse().unwrap()
}
