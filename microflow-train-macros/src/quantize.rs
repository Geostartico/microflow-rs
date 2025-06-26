use byterepr::ByteReprNum;
use nalgebra::Scalar;
use proc_macro2::TokenStream;
use quote::ToTokens;
use simba::scalar::SubsetOf;
use syn::ItemStruct;

/// Represents the trait to constrain a type to be quantized and tokenized.
pub(crate) trait TokenQuantized:
    Scalar + ByteReprNum + ToTokens + SubsetOf<i32> + SubsetOf<f32> + SubsetOf<i64>
{
}
//trait for layers to add their parameters
pub(crate) trait AddAttributes : ToTokens
{
    fn add_attrs(&self, attrs: &mut ItemStruct);
    fn define_members(&self, definitions: &mut TokenStream);
}

impl<T: Scalar + ByteReprNum + ToTokens + SubsetOf<i32> + SubsetOf<f32> + SubsetOf<i64>>
    TokenQuantized for T
{
}
