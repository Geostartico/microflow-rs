use crate::{
    activation::FusedActivation,
    buffer::Buffer4D,
    quantize::{quantize, Trainable},
    tensor::{Tensor4D, TensorViewPadding},
    update_layer::get_input_index,
};
use core::array;
use nalgebra::SMatrix;
pub fn gradient_average_pool<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTER_ROWS: usize,
    const FILTER_COLS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    outputs: Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS, 1>,
    output_grad: Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS>,
    _filter_shape: (nalgebra::Const<FILTER_ROWS>, nalgebra::Const<FILTER_COLS>),
    activation: FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let normalization_param = 1f32;
    let mut accum: Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| array::from_fn(|_| 0i32)));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            let coord = get_input_index(
                FILTER_ROWS,
                FILTER_COLS,
                (output_row, output_col),
                padding,
                strides,
            );
            for output_batch in 0..INPUT_CHANS {
                let val = outputs.buffer[0][(output_row, output_col)][output_batch]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                for filter_row in 0..FILTER_ROWS {
                    if (coord.0 + filter_row as i32) < 0 {
                        continue;
                    }
                    for filter_col in 0..FILTER_COLS {
                        if (coord.1 + filter_col as i32) < 0 {
                            continue;
                        }
                        let cur_coord = (
                            (coord.0 + filter_row as i32) as usize,
                            (coord.1 + filter_col as i32) as usize,
                        );
                        accum[0][cur_coord][output_batch] = accum[0][cur_coord][output_batch]
                            .saturating_add(output_grad[0][(output_row, output_col)][output_batch]);
                    }
                }
            }
        }
    }
    accum.map(|batch| {
        batch.map(|channels| channels.map(|ch| (ch as f32 / (normalization_param)).round() as i32))
    })
}
