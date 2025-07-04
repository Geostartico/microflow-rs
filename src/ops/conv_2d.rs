use core::array;
use libm::roundf;

use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::{Tensor4D, TensorView, TensorViewPadding};

pub struct Conv2DOptions {
    pub fused_activation: FusedActivation,
    pub view_padding: TensorViewPadding,
    pub strides: (usize, usize),
}

/// Performs the Conv2D operation.
/// Returns a 4-dimensional output tensor containing the result of the operation.
/// Takes ownership of the Tensor struct, is useful for memory optimization purposes.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor
/// * `filters` - The 4-dimensional tensor representing the filters of the operator
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as an [`Conv2DOptions`] struct
/// * `constants` - Constant values coming from the pre-processing phase
///
pub fn conv_2d<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTERS_BATCHES: usize,
    const FILTERS_ROWS: usize,
    const FILTERS_COLS: usize,
    const FILTERS_QUANTS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    filters: &Tensor4D<T, FILTERS_BATCHES, FILTERS_ROWS, FILTERS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: Conv2DOptions,
    constants: (
        Buffer2D<f32, FILTERS_BATCHES, 1>,
        Buffer2D<f32, FILTERS_QUANTS, 1>,
    ),
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_BATCHES, 1> {
    return conv_2d_borrow(&input, filters, output_scale, output_zero_point, options, constants)
}
/// Performs the Conv2D operation.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor
/// * `filters` - The 4-dimensional tensor representing the filters of the operator
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as an [`Conv2DOptions`] struct
/// * `constants` - Constant values coming from the pre-processing phase
///
pub fn conv_2d_borrow<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTERS_BATCHES: usize,
    const FILTERS_ROWS: usize,
    const FILTERS_COLS: usize,
    const FILTERS_QUANTS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    filters: &Tensor4D<T, FILTERS_BATCHES, FILTERS_ROWS, FILTERS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: Conv2DOptions,
    constants: (
        Buffer2D<f32, FILTERS_BATCHES, 1>,
        Buffer2D<f32, FILTERS_QUANTS, 1>,
    ),
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_BATCHES, 1> {
    let output = [Buffer2D::from_fn(|i, j| {
        // Extract the view using the view extraction algorithm
        let view: TensorView<T, FILTERS_ROWS, FILTERS_COLS, INPUT_CHANS> =
            input.view((i, j), 0, options.view_padding, options.strides);
        // Perform the convolution for each filter batch
        array::from_fn(|b| {
            let input_zero_point = i32::from_subset(&input.zero_point[0]);
            let filters_zero_point = i32::from_subset(
                &filters
                    .zero_point
                    .get(b)
                    .copied()
                    .unwrap_or(filters.zero_point[0]),
            );
            let x = (
                // Perform the dot product between the input region and the filter
                view.buffer.zip_fold(&filters.buffer[b], 0i32, |acc, v, f| {
                    acc + v
                        .iter()
                        .zip(f.iter())
                        .map(|(e1, e2)| i32::from_subset(e1) * i32::from_subset(e2))
                        .sum::<i32>()
                }),
                // Perform the 3-dimensional component-sum of the view
                view.buffer.fold(0i32, |acc, a| {
                    acc + a.iter().fold(0i32, |acc, e| acc + i32::from_subset(e))
                }) * filters_zero_point,
            );
            // Elaborate the constants
            let constants = (
                constants.0,
                constants.1,
                input_zero_point
                    * filters.buffer[b].zip_fold(&view.mask, 0i32, |acc, f, m| {
                        if m {
                            acc + f.iter().fold(0i32, |acc, e| acc + i32::from_subset(e))
                        } else {
                            acc
                        }
                    }),
                view.len as i32 * INPUT_CHANS as i32 * input_zero_point * filters_zero_point,
            );
            // Combine the constant values and the variants to obtain the output
            let y = T::from_superset_unchecked(&roundf(
                f32::from_subset(&output_zero_point[0])
                    + constants.0[b]
                    + constants.1.get(b).copied().unwrap_or(constants.1[0])
                        * f32::from_subset(&(x.0 - x.1 - constants.2 + constants.3)),
            ));
            // Apply the fused activation function (if any)
            match options.fused_activation {
                FusedActivation::None => y,
                FusedActivation::Relu => relu(y, output_zero_point[0]),
                FusedActivation::Relu6 => relu6(y, output_scale[0], output_zero_point[0]),
            }
        })
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use crate::tensor::Tensor2D;

    use super::*;

    const INPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4],  [5,  6];
            [7, 8], [9, 10], [11, 12]
        ]],
        scale: [0.13],
        zero_point: [14],
    };
    const FILTERS: Tensor4D<i8, 2, 2, 3, 2, 2> = Tensor4D {
        buffer: [
            matrix![
                [15, 16], [17, 18], [19, 20];
                [21, 22], [23, 24], [25, 26]
            ],
            matrix![
                [27, 28], [29, 30], [31, 32];
                [33, 34], [35, 36], [37, 38]
            ],
        ],
        scale: [0.39, 0.40],
        zero_point: [41, 42],
    };
    const _BIASES: Tensor2D<i32, 2, 1, 2> = Tensor2D {
        buffer: matrix![
            43;
            44
        ],
        scale: [0.45, 0.46],
        zero_point: [47, 48],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.49];
    const OUTPUT_ZERO_POINT: [i8; 1] = [50];
    const OPTIONS: Conv2DOptions = Conv2DOptions {
        fused_activation: FusedActivation::None,
        view_padding: TensorViewPadding::Same,
        strides: (1, 1),
    };
    const CONSTANTS: (Buffer2D<f32, 2, 1>, Buffer2D<f32, 2, 1>) = (
        matrix![-3.673_469_4; -3.755_102],
        matrix![0.103_469_39; 0.106_122_45],
    );
    const OUTPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [127, 116], [127, 127], [127, 113];
            [98,  74],  [114, 84],  [82,  67]
        ]],
        scale: [0.49],
        zero_point: [50],
    };

    #[test]
    fn conv_2d_layer() {
        assert_eq!(
            conv_2d(
                INPUT,
                &FILTERS,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                OPTIONS,
                CONSTANTS,
            ),
            OUTPUT
        );
    }
}
