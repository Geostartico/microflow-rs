//! [![crates.io](https://img.shields.io/crates/v/microflow-macros)](https://crates.io/crates/microflow-macros)
//! [![docs.rs](https://img.shields.io/docsrs/microflow-macros)](https://docs.rs/microflow-macros)
//! [![github](https://img.shields.io/github/actions/workflow/status/matteocarnelos/microflow-rs/cargo.yml?branch=main)](https://github.com/matteocarnelos/microflow-rs/actions/workflows/cargo.yml)
//!
//! Macro crate of the [MicroFlow](https://github.com/matteocarnelos/microflow-rs) inference engine, namely, the MicroFlow compiler.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};
use quantize::TrainToTokens;
use std::fs;

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::{parse_macro_input, ItemStruct};

use crate::tflite_flatbuffers::tflite::TensorType;
use ops::*;
use structmeta::StructMeta;
use syn::{LitInt, LitStr};
use tflite_flatbuffers::tflite::{root_as_model, BuiltinOperator};

mod activation;
mod buffer;
mod ops;
mod quantize;
mod tensor;
#[path = "../flatbuffers/tflite_generated.rs"]
#[allow(unused_imports)]
#[allow(clippy::all)]
mod tflite_flatbuffers;

#[derive(StructMeta)]
struct Args {
    #[struct_meta(unnamed)]
    path: LitStr,
    #[struct_meta(unnamed)]
    num_train_layers: LitInt,
}

/// The entry point of MicroFlow.
/// This attribute-like procedural macro can be placed on `structs` to implement the `predict()`
/// function based on the given model.
/// The macro takes as input the path of the model, which must be in the TensorFlow Lite format
/// (`.tflite`).
#[proc_macro_error]
#[proc_macro_attribute]
pub fn model(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as Args);
    let item = parse_macro_input!(item as ItemStruct);
    let mut item = item.clone();

    let buf = fs::read(args.path.value()).unwrap_or_else(|_| {
        abort_call_site!(
            "couldn't find '{}', please provide a valid path",
            &args.path.value()
        )
    });
    let model = root_as_model(&buf).unwrap_or_else(|_| {
        abort_call_site!("invalid model, please provide a valid TensorFlow Lite model")
    });
    let layers_to_train: usize = args.num_train_layers.base10_parse().unwrap();

    let subgraph = model.subgraphs().unwrap().get(0);
    let tensors = subgraph.tensors().unwrap();
    let buffers = model.buffers().unwrap();

    let input = tensors.get(subgraph.inputs().unwrap().get(0) as usize);
    let mut input_shape: Vec<_> = input.shape().unwrap().iter().map(|e| e as usize).collect();
    if input_shape.len() == 1 {
        input_shape.insert(0, 1);
    }
    let input_type = match input.type_() {
        TensorType::INT8 => quote!(i8),
        TensorType::UINT8 => quote!(u8),
        _ => unimplemented!(),
    };
    let input_tensor = match input_shape.len() {
        2 => quote!(Tensor2D),
        4 => quote!(Tensor4D),
        _ => unimplemented!(),
    };
    let input_buffer = match input_shape.len() {
        2 => quote!(Buffer2D),
        4 => quote!(Buffer4D),
        _ => unimplemented!(),
    };
    let input_scale: Vec<_> = input
        .quantization()
        .unwrap()
        .scale()
        .unwrap()
        .iter()
        .map(|e| e.to_token_stream())
        .collect();
    let input_zero_point: Vec<_> = match input.type_() {
        TensorType::INT8 => input
            .quantization()
            .unwrap()
            .zero_point()
            .unwrap()
            .iter()
            .map(|e| (e as i8).to_token_stream())
            .collect(),
        TensorType::UINT8 => input
            .quantization()
            .unwrap()
            .zero_point()
            .unwrap()
            .iter()
            .map(|e| (e as u8).to_token_stream())
            .collect(),
        _ => unimplemented!(),
    };

    let operators = subgraph.operators().unwrap();
    let mut layers = TokenStream2::new();
    let mut layers_train = TokenStream2::new();
    let mut new_declarations = TokenStream2::new();
    let mut backward = TokenStream2::new();
    for (index, operator) in operators.iter().enumerate() {
        let layer_num = index as i32 - (operators.len() as i32 - layers_to_train as i32);
        if layer_num < 0 {
            let layer: Box<dyn ToTokens> = match BuiltinOperator(
                model
                    .operator_codes()
                    .unwrap()
                    .get(operator.opcode_index() as usize)
                    .deprecated_builtin_code() as i32,
            ) {
                BuiltinOperator::FULLY_CONNECTED => {
                    fully_connected::parse(operator, tensors, buffers, index)
                }
                BuiltinOperator::DEPTHWISE_CONV_2D => {
                    depthwise_conv_2d::parse(operator, tensors, buffers, index)
                }
                BuiltinOperator::CONV_2D => conv_2d::parse(operator, tensors, buffers, index),
                BuiltinOperator::AVERAGE_POOL_2D => average_pool_2d::parse(operator, tensors),
                BuiltinOperator::SOFTMAX => softmax::parse(operator, tensors),
                BuiltinOperator::RESHAPE => Box::new(reshape::parse(operator, tensors)),
                unsupported_op => abort_call_site!("unsupported operator: {:?}", unsupported_op),
            };
            layer.to_tokens(&mut layers);
            layer.to_tokens(&mut layers_train)
        } else {
            let mut layer: Box<dyn TrainToTokens> = match BuiltinOperator(
                model
                    .operator_codes()
                    .unwrap()
                    .get(operator.opcode_index() as usize)
                    .deprecated_builtin_code() as i32,
            ) {
                BuiltinOperator::FULLY_CONNECTED => {
                    fully_connected::parse_indexed(operator, tensors, buffers, index, layer_num)
                }
                BuiltinOperator::DEPTHWISE_CONV_2D => {
                    depthwise_conv_2d::parse_indexed(operator, tensors, buffers, index, layer_num)
                }
                BuiltinOperator::CONV_2D => {
                    conv_2d::parse_indexed(operator, tensors, buffers, index, layer_num)
                }
                BuiltinOperator::AVERAGE_POOL_2D => {
                    average_pool_2d::parse_indexed(operator, tensors, layer_num)
                }
                BuiltinOperator::SOFTMAX => softmax::parse_indexed(operator, tensors, layer_num),
                BuiltinOperator::RESHAPE => reshape::parse_indexed(operator, tensors, layer_num),
                unsupported_op => abort_call_site!("unsupported operator: {:?}", unsupported_op),
            };
            layer.add_attrs(&mut item);
            layer.define_members(&mut new_declarations);
            layer.train_ops(&mut backward);
            layer.to_tokens(&mut layers);
            layer.switch_train();
            layer.to_tokens(&mut layers_train);
        }
    }

    let ident = &item.ident;
    let output = tensors.get(subgraph.outputs().unwrap().get(0) as usize);
    let mut output_shape: Vec<_> = output.shape().unwrap().iter().map(|e| e as usize).collect();
    if output_shape.len() == 1 {
        output_shape.insert(0, 1);
    }
    let output_type = match output.type_() {
        TensorType::INT8 => quote!(i8),
        TensorType::UINT8 => quote!(u8),
        _ => unimplemented!(),
    };
    let output_tensor = match output_shape.len() {
        2 => quote!(Tensor2D),
        4 => quote!(Tensor4D),
        _ => unimplemented!(),
    };
    let output_buffer = match output_shape.len() {
        2 => quote!(Buffer2D),
        4 => quote!(Buffer4D),
        _ => unimplemented!(),
    };

    let output_ident = format_ident!("input{}", layers_to_train - 1);
    let ts = quote! {
        #item
        impl #ident {
            pub fn new() -> Self {
                #ident {
                    #new_declarations
                }
            }
            pub fn predict(&self, input: microflow::buffer::#input_buffer<f32, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::quantize(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                self.predict_inner(input).dequantize()
            }

            pub fn predict_quantized(&self, input: microflow::buffer::#input_buffer<#input_type, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::new(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                self.predict_inner(input).dequantize()
            }

            fn predict_inner(&self, input: microflow::tensor::#input_tensor<#input_type, #(#input_shape),*, 1usize>) -> microflow::tensor::#output_tensor<#output_type, #(#output_shape),*, 1usize> {
                #layers
                input
            }
            pub fn predict_train(&mut self, input: microflow::buffer::#input_buffer<f32, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::quantize(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                self.predict_inner(input).dequantize()
            }

            pub fn predict_quantized_train(&mut self, input: microflow::buffer::#input_buffer<#input_type, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::new(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                self.predict_inner(input).dequantize()
            }

            fn predict_inner_train(&mut self, input: microflow::tensor::#input_tensor<#input_type, #(#input_shape),*, 1usize>, learning_rate: f32) -> microflow::tensor::#output_tensor<#output_type, #(#output_shape),*, 1usize> {
                #layers_train
                let backward_gradient = #output_ident.buffer;
                #backward
                #output_ident
            }
        }
    };

    fs::write("target/microflow-expansion.rs", ts.to_string()).ok();

    ts.into()
}
