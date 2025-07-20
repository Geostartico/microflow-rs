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

/// Represents the tokenized version of the `Conv2D` operator.
pub(crate) struct TokenConv2D<T: TokenQuantized> {
    pub(crate) filters: TokenTensor4D<T>,
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) view_padding: TokenTensorViewPadding,
    pub(crate) strides: (usize, usize),
    pub(crate) constants: (TokenBuffer2D<f32>, TokenBuffer2D<f32>),
    pub(crate) constants_gradient: (TokenBuffer2D<f32>, TokenBuffer2D<f32>),
    pub(crate) scale_bias: Vec<f32>,
    pub(crate) index: usize,
    pub(crate) layer_index: i32,
    pub(crate) train: bool,
}

/// Parses the [`TokenConv2D`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
/// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
/// * `index` - The operator index
/// * `layer_index` - index of the layer, used for the training pipeline
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
        TensorType::INT8 => Box::new(TokenConv2D::<i8>::new(
            operator,
            tensors,
            buffers,
            index,
            layer_index,
        )),
        TensorType::UINT8 => Box::new(TokenConv2D::<u8>::new(
            operator,
            tensors,
            buffers,
            index,
            layer_index,
        )),
        _ => unimplemented!(),
    }
}

/// Parses the [`TokenConv2D`] struct from the given operator.
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
        TensorType::INT8 => Box::new(TokenConv2D::<i8>::new(
            operator, tensors, buffers, index, -1,
        )),
        TensorType::UINT8 => Box::new(TokenConv2D::<u8>::new(
            operator, tensors, buffers, index, -1,
        )),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenConv2D<T> {
    /// Builds the [`TokenConv2D`] operator from the given model operator and tensors.
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
        let filters =
            TokenTensor4D::from_buffered_tensor(tensors.get(inputs.get(1) as usize), buffers);
        let biases =
            TokenTensor2D::from_buffered_tensor(tensors.get(inputs.get(2) as usize), buffers);
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator.builtin_options_as_conv_2_doptions().unwrap();
        let constants = Self::preprocess(&input, &filters, &biases, &output);
        let constants_gradient = Self::gen_zero_constants(&filters);
        Self {
            filters,
            output,
            fused_activation: options.fused_activation_function().into(),
            view_padding: options.padding().into(),
            strides: (options.stride_h() as usize, options.stride_w() as usize),
            constants,
            constants_gradient,
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
    /// * `filters` - The filters of the operator as a [`TokenTensor2D`]
    /// * `biases` - The biases of the operator as a [`TokenTensor2D`]
    /// * `output` - The output of the operator as a [`TokenTensor2D`]
    ///
    fn preprocess(
        input: &TokenTensor4D<T>,
        filters: &TokenTensor4D<T>,
        biases: &TokenTensor2D<i32>,
        output: &TokenTensor4D<T>,
    ) -> (TokenBuffer2D<f32>, TokenBuffer2D<f32>) {
        (
            TokenBuffer2D::from(DMatrix::from_fn(filters.shape[0], 1, |b, _| {
                biases.scale.get(b).copied().unwrap_or(biases.scale[0]) / output.scale[0]
                    * (biases.buffer[b]
                        - biases
                            .zero_point
                            .get(b)
                            .copied()
                            .unwrap_or(biases.zero_point[0])) as f32
            })),
            TokenBuffer2D::from(DMatrix::from_fn(filters.scale.len(), 1, |b, _| {
                input.scale[0] * filters.scale[b] / output.scale[0]
            })),
        )
    }
    fn gen_zero_constants(filters: &TokenTensor4D<T>) -> (TokenBuffer2D<f32>, TokenBuffer2D<f32>) {
        (
            TokenBuffer2D::from(DMatrix::from_fn(filters.shape[0], 1, |_, _| 0f32)),
            TokenBuffer2D::from(DMatrix::from_fn(filters.scale.len(), 1, |_, _| 0f32)),
        )
    }
}
impl<T: TokenQuantized> TrainToTokens for TokenConv2D<T> {
    fn add_attrs(&self, attrs: &mut ItemStruct) {
        let filters_ident = format_ident!("filters{}", self.layer_index as usize);
        let filters_gradient_ident = format_ident!("filters{}_gradient", self.layer_index as usize);
        let filters_type = self.filters.type_tokens();
        let filters_shape = &self.filters.shape;
        let filters_shape_0 = filters_shape[0];
        let filters_shape_1 = filters_shape[1];
        let filters_shape_2 = filters_shape[2];
        let filters_shape_3 = filters_shape[3];
        let constants_field_name = format_ident!("constants{}", self.layer_index as usize);
        let constants_gradient_field_name =
            format_ident!("constants{}_gradient", self.layer_index as usize);
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
        let constants_field_gradient: syn::Field = syn::parse_quote! {
            #constants_gradient_field_name: #constants_field_type
        };
        let filters_field_gradient: syn::Field = syn::parse_quote! {
            #filters_gradient_ident: Buffer4D<i32,#filters_shape_0,#filters_shape_1,#filters_shape_2,#filters_shape_3>
        };
        match &mut attrs.fields {
            syn::Fields::Named(ref mut fields_named) => {
                fields_named.named.push(constants_field);
                fields_named.named.push(filters_field);
                fields_named.named.push(constants_field_gradient);
                fields_named.named.push(filters_field_gradient);
            }
            _ => panic!("add_fields only works with structs with named fields"),
        }
    }
    fn define_members(&self, declarations: &mut TokenStream2) {
        let constants_field_name = format_ident!("constants{}", self.layer_index as usize);
        let filters_ident = format_ident!("filters{}", self.layer_index as usize);
        let constants_gradient_field_name =
            format_ident!("constants{}_gradient", self.layer_index as usize);
        let filters_gradient_ident = format_ident!("filters{}_gradient", self.layer_index as usize);
        let filters = &self.filters;
        let (constants_0, constants_1) = &self.constants;
        let (constants_0_gradient, constants_1_gradient) = &self.constants_gradient;
        let ts = quote! {
            #filters_ident : #filters,
            #constants_field_name : (#constants_0, #constants_1),
            #filters_gradient_ident : array::from_fn(|_|SMatrix::from_fn(|_,_|array::from_fn(|_|0i32))),
            #constants_gradient_field_name : (#constants_0_gradient, #constants_1_gradient),
        };
        ts.to_tokens(declarations);
    }
    fn switch_train(&mut self) {
        self.train = !self.train;
    }
    fn train_ops(&self, backward: &mut TokenStream2) {
        let weights_ident: syn::Expr = {
            let field_ident = format_ident!("filters{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let weights_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("filters{}_gradient", self.layer_index as usize);
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
        let constants_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}_gradient", self.layer_index as usize);
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
            let (backward_gradient, weight_gradient) = microflow::gradient_conv_2d::update_grad_conv_2d(
                &#input_ident,
                &#weights_ident,
                &mut #weights_gradient_ident,
                & #constants_ident,
                &mut #constants_gradient_ident,
                #output_ident,
                backward_gradient,
                #activation,
                (#strides_0,#strides_1),
                #view_padding,
                [#(#bias_scale),*],
                learning_rate,
            );
            // println!("gradient weights: {}",weight_gradient.iter().fold(String::new(), |accarr, batch|accarr + &batch.map(|el|el.iter().fold(String::new(),|sum, el1|sum +" "+ &el1.to_string())).to_string()));
            // println!("gradient input: {}",backward_gradient[0].map(|el|el.iter().fold(String::new(),|sum, el1|sum +" " +&el1.to_string())));
            // println!("mean gradient conv: {}",weight_gradient[0].map(|el|el.iter().fold(0f32,|sum, el1|sum+(*el1 as f32).abs() / el.len() as f32)).mean());
        };
        let mut ts = TokenStream2::new();
        prepend.to_tokens(&mut ts);
        ts.extend(backward.clone());
        *backward = ts;
    }
    fn update_ops(&self, updates: &mut TokenStream2) {
        let weights_ident: syn::Expr = {
            let field_ident = format_ident!("filters{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let weights_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("filters{}_gradient", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let constants_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let constants_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}_gradient", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let update = quote! {
            microflow::update_layer::update_weights_4D(
                &mut #weights_ident,
                &#weights_gradient_ident,
                batch_size,
                learning_rate,
            );
            microflow::update_layer::update_weights_2D_float(
                &mut #constants_ident.0,
                &#constants_gradient_ident.0,
                batch_size,
                learning_rate,
            );
            #weights_gradient_ident = array::from_fn(|_|SMatrix::from_fn(|_,_|array::from_fn(|_|0i32)));
            #constants_gradient_ident.0 = SMatrix::zeros();
            // println!("gradient weights: {}",weight_gradient.iter().fold(String::new(), |accarr, batch|accarr + &batch.map(|el|el.iter().fold(String::new(),|sum, el1|sum +" "+ &el1.to_string())).to_string()));
            // println!("gradient input: {}",backward_gradient[0].map(|el|el.iter().fold(String::new(),|sum, el1|sum +" " +&el1.to_string())));
            // println!("mean gradient conv: {}",weight_gradient[0].map(|el|el.iter().fold(0f32,|sum, el1|sum+(*el1 as f32).abs() / el.len() as f32)).mean());
        };
        update.to_tokens(updates);
    }
}

impl<T: TokenQuantized> ToTokens for TokenConv2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let filters_ident: syn::Expr = if self.layer_index >= 0 {
            let field_ident = format_ident!("filters{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        } else {
            let field_ident = format_ident!("filters{}", self.index);
            parse_quote!(#field_ident)
        };
        let constants_ident: syn::Expr = if self.layer_index >= 0 {
            let field_ident = format_ident!("constants{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        } else {
            let (constants_0, constants_1) = &self.constants;
            parse_quote!((#constants_0, #constants_1))
        };
        let filters_type = self.filters.type_tokens();
        let filters = &self.filters;
        let filters_declaration = if self.layer_index < 0 {
            quote! {const #filters_ident: #filters_type = #filters;}
        } else {
            quote! {}
        };
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let (constants_0, constants_1) = &self.constants;
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
            parse_quote!(microflow::ops::conv_2d_borrow)
        } else {
            parse_quote!(microflow::ops::conv_2d)
        };
        let ts = quote! {
            #filters_declaration
            let #output_name: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                #func_name(
                    #reference_tok (#input_name),
                    &#filters_ident,
                    [#(#output_scale),*],
                    [#(#output_zero_point),*],
                    microflow::ops::Conv2DOptions {
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

    fn setup() -> TokenConv2D<i8> {
        TokenConv2D {
            filters: TokenTensor4D {
                buffer: TokenBuffer4D::from(vec![
                    dmatrix![
                        vec![1, 2], vec![3,  4],  vec![5,  6];
                        vec![7, 8], vec![9,  10], vec![11, 12]
                    ],
                    dmatrix![
                        vec![13, 14], vec![15, 16], vec![17, 18];
                        vec![19, 20], vec![21, 22], vec![23, 24]
                    ],
                ]),
                shape: vec![2, 2, 3, 2],
                scale: vec![0.25, 0.26],
                zero_point: vec![27, 28],
            },
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.29],
                zero_point: vec![30],
            },
            fused_activation: TokenFusedActivation::Relu6,
            view_padding: TokenTensorViewPadding::Same,
            strides: (1, 1),
            constants: (
                TokenBuffer2D::from(dmatrix![31., 32.]),
                TokenBuffer2D::from(dmatrix![33., 34.]),
            ),
            index: 0,
            layer_index: -1,
        }
    }

    #[test]
    fn conv_2d_preprocess() {
        let layer = setup();
        let input = TokenTensor4D {
            buffer: TokenBuffer4D::new(),
            shape: vec![1, 2, 3, 2],
            scale: vec![0.35],
            zero_point: vec![36],
        };
        let biases = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                37;
                38
            ]),
            shape: vec![2, 1],
            scale: vec![0.39, 0.40],
            zero_point: vec![41, 42],
        };
        let constants = TokenConv2D::preprocess(&input, &layer.filters, &biases, &layer.output);
        assert_eq!(constants.0 .0, Some(dmatrix![-5.37931; -5.5172415]));
        assert_eq!(constants.1 .0, Some(dmatrix![0.30172414; 0.3137931]));
    }

    #[test]
    fn conv_2d_to_tokens() {
        let layer = setup();
        let filters = &layer.filters;
        let fused_activation = layer.fused_activation;
        let view_padding = layer.view_padding;
        let (constants_0, constants_1) = &layer.constants;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                const filters_0: microflow::tensor::Tensor4D<i8, 2usize, 2usize, 3usize, 2usize, 2usize> = #filters;
                let input: microflow::tensor::Tensor4D<_, 1usize, 2usize, 3usize, 2usize, 1usize> =
                    microflow::ops::conv_2d(
                        input,
                        &filters_0,
                        [0.29f32],
                        [30i8],
                        microflow::ops::Conv2DOptions {
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
