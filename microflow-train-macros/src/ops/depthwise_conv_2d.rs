use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::{TokenQuantized, TrainToTokens};
use crate::tensor::{TokenTensor2D, TokenTensor4D, TokenTensorViewPadding};
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::DMatrix;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::{parse_quote, ItemStruct};

/// Represents the tokenized version of the `DepthwiseConv2D` operator.
pub(crate) struct TokenDepthwiseConv2D<T: TokenQuantized> {
    pub(crate) weights: TokenTensor4D<T>,
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) view_padding: TokenTensorViewPadding,
    pub(crate) strides: (usize, usize),
    pub(crate) constants: (TokenBuffer2D<f32>, TokenBuffer2D<f32>),
    pub(crate) index: usize,
    pub(crate) layer_index: i32,
    pub(crate) scale_bias: Vec<f32>,
    pub(crate) train: bool,
}

/// Parses the [`TokenDepthwiseConv2D`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
/// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
/// * `index` - The operator index
///
pub(crate) fn parse_indexed(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
    layer_index: i32,
) -> Box<dyn TrainToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenDepthwiseConv2D::<i8>::new(
            operator,
            tensors,
            buffers,
            index,
            layer_index,
        )),
        TensorType::UINT8 => Box::new(TokenDepthwiseConv2D::<u8>::new(
            operator,
            tensors,
            buffers,
            index,
            layer_index,
        )),
        _ => unimplemented!(),
    }
}

/// Parses the [`TokenDepthwiseConv2D`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
/// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
/// * `index` - The operator index
///
pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenDepthwiseConv2D::<i8>::new(
            operator, tensors, buffers, index, -1,
        )),
        TensorType::UINT8 => Box::new(TokenDepthwiseConv2D::<u8>::new(
            operator, tensors, buffers, index, -1,
        )),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenDepthwiseConv2D<T> {
    /// Builds the [`TokenDepthwiseConv2D`] operator from the given model operator and tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    /// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
    /// * `index` - The operator index
    ///
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        index: usize,
        layer_index: i32,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor4D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let weights =
            TokenTensor4D::from_buffered_tensor(tensors.get(inputs.get(1) as usize), buffers);
        let biases =
            TokenTensor2D::from_buffered_tensor(tensors.get(inputs.get(2) as usize), buffers);
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator
            .builtin_options_as_depthwise_conv_2_doptions()
            .unwrap();
        let constants = Self::preprocess(&input, &weights, &biases, &output);
        Self {
            weights,
            output,
            fused_activation: options.fused_activation_function().into(),
            view_padding: options.padding().into(),
            strides: (options.stride_h() as usize, options.stride_w() as usize),
            constants,
            index,
            layer_index,
            train: false,
            scale_bias: biases.scale,
        }
    }

    /// Pre-processes the operator, returning the tuple of constants.
    ///
    /// # Arguments
    /// * `input` - The input of the operator as a [`TokenTensor2D`]
    /// * `weights` - The weights of the operator as a [`TokenTensor2D`]
    /// * `biases` - The biases of the operator as a [`TokenTensor2D`]
    /// * `output` - The output of the operator as a [`TokenTensor2D`]
    ///
    fn preprocess(
        input: &TokenTensor4D<T>,
        weights: &TokenTensor4D<T>,
        biases: &TokenTensor2D<i32>,
        output: &TokenTensor4D<T>,
    ) -> (TokenBuffer2D<f32>, TokenBuffer2D<f32>) {
        (
            TokenBuffer2D::from(DMatrix::from_fn(weights.shape[3], 1, |c, _| {
                biases.scale.get(c).copied().unwrap_or(biases.scale[0]) / output.scale[0]
                    * (biases.buffer[c]
                        - biases
                            .zero_point
                            .get(c)
                            .copied()
                            .unwrap_or(biases.zero_point[0])) as f32
            })),
            TokenBuffer2D::from(DMatrix::from_fn(weights.scale.len(), 1, |c, _| {
                input.scale[0] * weights.scale[c] / output.scale[0]
            })),
        )
    }
}

impl<T: TokenQuantized> TrainToTokens for TokenDepthwiseConv2D<T> {
    fn add_attrs(&self, attrs: &mut ItemStruct) {
        let filters_ident = format_ident!("weights{}", self.layer_index as usize);
        let filters_type = self.weights.type_tokens();
        let constants_field_name = format_ident!("constants{}", self.layer_index as usize);
        let dim00 = self.constants.0.shape().0;
        let dim01 = self.constants.0.shape().1;
        let dim10 = self.constants.1.shape().0;
        let dim11 = self.constants.1.shape().1;
        let constants_field_type: syn::Type = parse_quote! {
            ( Buffer2D<f32, #dim00, #dim01>, Buffer2D<f32, #dim10, #dim11>,)
        };

        let constants_field: syn::Field = syn::parse_quote! {
            #constants_field_name: #constants_field_type
        };
        let filters_field: syn::Field = syn::parse_quote! {
            #filters_ident: #filters_type
        };
        match &mut attrs.fields {
            syn::Fields::Named(ref mut fields_named) => {
                fields_named.named.push(constants_field);
                fields_named.named.push(filters_field);
            }
            _ => panic!("add_fields only works with structs with named fields"),
        }
    }
    fn define_members(&self, declarations: &mut TokenStream2) {
        let constants_field_name = format_ident!("constants{}", self.layer_index as usize);
        let filters_ident = format_ident!("weights{}", self.layer_index as usize);
        let filters = &self.weights;
        let (constants_0, constants_1) = &self.constants;
        let ts = quote! {
            #filters_ident : #filters,
            #constants_field_name : (#constants_0, #constants_1),
        };
        ts.to_tokens(declarations);
    }
    fn switch_train(&mut self) {
        self.train = !self.train;
    }
    fn train_ops(&self, backward: &mut TokenStream2) {
        let weights_ident: syn::Expr = {
            let field_ident = format_ident!("weights{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let output_ident = if self.layer_index >= 0 {
            format_ident!("input{}", self.layer_index as usize)
        } else {
            format_ident!("input")
        };
        let input_ident = if self.layer_index > 0 {
            format_ident!("input{}", (self.layer_index - 1) as usize)
        } else {
            format_ident!("input")
        };
        let constants_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let bias_scale = self
            .scale_bias
            .iter()
            .map(|x| quote! { #x })
            .collect::<Vec<_>>();
        let prepend = quote! {
            let backward_gradient = microflow::gradient_depthwise_conv_2d::update_grad_depthwise_conv_2d(
                &#input_ident,
                &mut #weights_ident,
                &mut #constants_ident,
                #output_ident,
                backward_gradient,
                #activation,
                (#strides_0,#strides_1),
                #view_padding,
                [#(#bias_scale),*],
                learning_rate,
            );
        };
        let mut ts = TokenStream2::new();
        prepend.to_tokens(&mut ts);
        ts.extend(backward.clone());
        *backward = ts;
    }
}
impl<T: TokenQuantized> ToTokens for TokenDepthwiseConv2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let weights_ident: syn::Expr = if self.layer_index >= 0 {
            let field_ident = format_ident!("weights{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        } else {
            let field_ident = format_ident!("weights{}", self.index);
            parse_quote!(#field_ident)
        };
        let constants_ident: syn::Expr = if self.layer_index >= 0 {
            let field_ident = format_ident!("constants{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        } else {
            let (constants_0, constants_1) = &self.constants;
            parse_quote!((#constants_0, #constants_1))
        };
        let weights_type = self.weights.type_tokens();
        let weights = &self.weights;
        let weights_declaration = if self.layer_index < 0 {
            quote! {const #weights_ident: #weights_type = #weights;}
        } else {
            quote! {}
        };
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let reference_tok = if self.layer_index >= 0 && self.train {
            quote! {&}
        } else {
            quote! {}
        };
        let output_name = if self.layer_index >= 0 && self.train {
            format_ident!("input{}", self.layer_index as usize)
        } else {
            format_ident!("input")
        };
        let input_name = if self.layer_index > 0 && self.train {
            format_ident!("input{}", (self.layer_index - 1) as usize)
        } else {
            format_ident!("input")
        };
        let func_name: syn::Path = if self.layer_index >= 0 && self.train {
            parse_quote!(microflow::ops::depthwise_conv_2d_borrow)
        } else {
            parse_quote!(microflow::ops::depthwise_conv_2d)
        };
        let ts = quote! {
            #weights_declaration
            let #output_name: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                #func_name(
                    #reference_tok (#input_name),
                    &#weights_ident,
                    [#(#output_scale),*],
                    [#(#output_zero_point),*],
                    microflow::ops::DepthwiseConv2DOptions {
                        fused_activation: #fused_activation,
                        view_padding: #view_padding,
                        strides: (#strides_0, #strides_1),
                    },
                    #constants_ident
            );
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{TokenBuffer2D, TokenBuffer4D};
    use nalgebra::dmatrix;

    fn setup() -> TokenDepthwiseConv2D<i8> {
        TokenDepthwiseConv2D {
            weights: TokenTensor4D {
                buffer: TokenBuffer4D::from(vec![dmatrix![
                    vec![1, 2], vec![3,  4],  vec![5,  6];
                    vec![7, 8], vec![9,  10], vec![11, 12]
                ]]),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.13, 0.14],
                zero_point: vec![15, 16],
            },
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.17],
                zero_point: vec![18],
            },
            fused_activation: TokenFusedActivation::Relu6,
            view_padding: TokenTensorViewPadding::Same,
            strides: (1, 1),
            constants: (
                TokenBuffer2D::from(dmatrix![19., 20.]),
                TokenBuffer2D::from(dmatrix![21., 22.]),
            ),
            index: 0,
            layer_index: -1,
            train: false,
        }
    }

    #[test]
    fn depthwise_conv_2d_preprocess() {
        let layer = setup();
        let input = TokenTensor4D {
            buffer: TokenBuffer4D::new(),
            shape: vec![1, 2, 3, 2],
            scale: vec![0.23],
            zero_point: vec![24],
        };
        let biases = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                25;
                26
            ]),
            shape: vec![2, 1],
            scale: vec![0.27, 0.28],
            zero_point: vec![29, 30],
        };
        let constants =
            TokenDepthwiseConv2D::preprocess(&input, &layer.weights, &biases, &layer.output);
        assert_eq!(constants.0 .0, Some(dmatrix![-6.3529415; -6.5882354]));
        assert_eq!(constants.1 .0, Some(dmatrix![0.17588235; 0.18941177]))
    }

    #[test]
    fn depthwise_conv_2d_to_tokens() {
        let layer = setup();
        let weights = &layer.weights;
        let fused_activation = layer.fused_activation;
        let view_padding = layer.view_padding;
        let (constants_0, constants_1) = &layer.constants;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                const weights_0: microflow::tensor::Tensor4D<i8, 1usize, 2usize, 3usize, 2usize, 2usize> = #weights;
                let input: microflow::tensor::Tensor4D<_, 1usize, 2usize, 3usize, 2usize, 1usize> =
                    microflow::ops::depthwise_conv_2d(
                        input,
                        &weights_0,
                        [0.17f32],
                        [18i8],
                        microflow::ops::DepthwiseConv2DOptions {
                            fused_activation: #fused_activation,
                            view_padding: #view_padding,
                            strides: (1usize, 1usize),
                        },
                        (#constants_0, #constants_1)
                );
            }.to_string()
        );
    }
}
