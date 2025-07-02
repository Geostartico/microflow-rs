use crate::{
    activation::FusedActivation,
    buffer::Buffer4D,
    quantize::{quantize, Trainable},
    tensor::{Tensor4D, TensorViewPadding},
    update_layer::get_input_index,
};
use core::array;
use nalgebra::{Const, SMatrix};
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
    output_grad: Buffer4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS>,
    _filter_shape: (nalgebra::Const<FILTER_ROWS>, nalgebra::Const<FILTER_COLS>),
    activation: FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| array::from_fn(|_| T::zero())));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
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
                let coord = get_input_index(
                    FILTER_ROWS,
                    FILTER_COLS,
                    (output_row, output_col),
                    padding,
                    strides,
                );
                for filter_row in 0..FILTER_ROWS {
                    if (coord.0 + filter_row as i32) < 0 {
                        continue;
                    }
                    for filter_col in 0..FILTER_COLS {
                        if (coord.1 + filter_col as i32) < 0 {
                            continue;
                        }
                        for filter_chans in 0..INPUT_CHANS {
                            let cur_coord = (
                                (coord.0 + filter_row as i32) as usize,
                                (coord.1 + filter_col as i32) as usize,
                            );
                            accum[output_batch][cur_coord][filter_chans] =
                                accum[output_batch][cur_coord][filter_chans].saturating_add(
                                    output_grad[0][(output_row, output_col)][output_batch],
                                );
                        }
                    }
                }
            }
        }
    }
    array::from_fn(|batch| {
        let scale =
            1f32 / (FILTER_ROWS as f32 * FILTER_COLS as f32 * outputs.scale[0] * input.scale[0]);
        SMatrix::from_fn(|i, j| {
            array::from_fn(|channel| {
                let tmp: f32 = T::to_superset(&accum[batch][(i, j)][channel]);
                T::from_superset(&(tmp * scale)).unwrap()
            })
        })
    })
}
