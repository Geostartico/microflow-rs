use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident,quote, ToTokens};

/// Represents the tokenized version of the `Softmax` operator.
pub(crate) struct TokenSoftmax<T: TokenQuantized> {
    pub(crate) output: TokenTensor2D<T>,
    pub(crate) layer_index: i32,
}

/// Parses the [`TokenSoftmax`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
///
pub(crate) fn parse_indexed(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    layer_index: i32
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenSoftmax::<i8>::new(operator, tensors, layer_index)),
        TensorType::UINT8 => Box::new(TokenSoftmax::<u8>::new(operator, tensors, layer_index)),
        _ => unimplemented!(),
    }
}

/// Parses the [`TokenSoftmax`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
///
pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenSoftmax::<i8>::new(operator, tensors, -1)),
        TensorType::UINT8 => Box::new(TokenSoftmax::<u8>::new(operator, tensors, -1)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenSoftmax<T> {
    /// Builds the [`TokenSoftmax`] operator from the given model operator and tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    ///
    pub(crate) fn new(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>, layer_index: i32) -> Self {
        let output = TokenTensor2D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        Self { output , layer_index}
    }
}

impl<T: TokenQuantized> ToTokens for TokenSoftmax<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let output_name = if self.layer_index >= 0 {format_ident!("input{}", self.layer_index as usize)} else {format_ident!("input")};
        let input_name = if self.layer_index > 0 {format_ident!("input{}", (self.layer_index - 1) as usize)} else {format_ident!("input")};
        let ts = quote! {
            let #output_name: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> =
                microflow::ops::softmax(#input_name, [#(#output_scale),*], [#(#output_zero_point),*]);
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer2D;

    fn setup() -> TokenSoftmax<i8> {
        TokenSoftmax {
            output: TokenTensor2D {
                buffer: TokenBuffer2D::new(),
                shape: vec![2, 3],
                scale: vec![0.3],
                zero_point: vec![4],
            },
            layer_index: -1
        }
    }

    #[test]
    fn softmax_to_tokens() {
        let layer = setup();
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor2D<_, 2usize, 3usize, 1usize> =
                    microflow::ops::softmax(input, [0.3f32], [4i8]);
            }
            .to_string()
        )
    }
}
