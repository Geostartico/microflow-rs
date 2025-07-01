use crate::{
    activation::FusedActivation,
    buffer::Buffer4D,
    quantize::{quantize, Trainable},
    tensor::{Tensor4D, TensorView, TensorViewPadding},
    update_layer::{get_input_index, update_weights_4D},
};
use core::array;
use nalgebra::{SMatrix, SVector};

pub fn update_grad_depthwise_conv_2d<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const FILTER_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &mut Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTER_QUANTS>,
    outputs: Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS, 1>,
    output_grad: Buffer4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
    learning_rate: f32,
) -> Buffer4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let grad_weight = grad_depthwise_conv_2d_weights(
        input,
        weights,
        &outputs,
        &output_grad,
        &activation,
        strides,
        padding,
    );
    update_weights_4D(weights, grad_weight, learning_rate);
    grad_depthwise_conv_2d_inputs(
        input,
        weights,
        &outputs,
        &output_grad,
        &activation,
        strides,
        padding,
    )
}
pub fn grad_depthwise_conv_2d_inputs<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const FILTER_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTER_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS, 1>,
    output_grad: &Buffer4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| [T::zero(); INPUT_CHANS]));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in (0..INPUT_ROWS).step_by(strides.0) {
        for output_col in (0..INPUT_COLS).step_by(strides.1) {
            for output_channel in (0..INPUT_CHANS).step_by(strides.1) {
                let val = outputs.buffer[0][(output_row, output_col)][output_channel]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                let coord = get_input_index(
                    WEIGHTS_ROWS,
                    WEIGHTS_COLS,
                    (output_row, output_col),
                    padding,
                    strides,
                );
                for filter_row in 0..WEIGHTS_ROWS {
                    if (coord.0 + filter_row as i32) < 0 {
                        continue;
                    }
                    for filter_col in 0..WEIGHTS_ROWS {
                        if (coord.1 + filter_col as i32) < 0 {
                            continue;
                        }
                        let cur_coord = (
                            (coord.0 + filter_row as i32) as usize,
                            (coord.1 + filter_col as i32) as usize,
                        );
                        accum[0][cur_coord][output_channel] = accum[0][cur_coord][output_channel]
                            .saturating_add(
                                input.buffer[0][cur_coord][output_channel]
                                    .saturating_sub(weights.zero_point[0]),
                            )
                            .saturating_mul(
                                &output_grad[0][(output_row, output_col)][output_channel],
                            );
                    }
                }
            }
        }
    }
    array::from_fn(|batch| {
        let filters_scale = weights
            .scale
            .get(batch)
            .copied()
            .unwrap_or(weights.scale[0]);
        let scale = filters_scale * input.scale[0];
        SMatrix::from_fn(|i, j| {
            array::from_fn(|channel| {
                let tmp: f32 = T::to_superset(&accum[batch][(i, j)][channel]);
                T::from_superset(&(tmp * scale)).unwrap()
            })
        })
    })
}
pub fn grad_depthwise_conv_2d_weights<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, WEIGHTS_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS, 1>,
    output_grad: &Buffer4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| [T::zero(); INPUT_CHANS]));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in (0..INPUT_ROWS).step_by(strides.0) {
        for output_col in (0..INPUT_COLS).step_by(strides.1) {
            for output_channel in (0..INPUT_CHANS).step_by(strides.1) {
                let val = outputs.buffer[0][(output_row, output_col)][output_channel]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                let view: TensorView<T, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> =
                    input.view((output_row, output_col), 0, padding, strides);
                for filter_row in 0..WEIGHTS_ROWS {
                    for filter_cols in 0..WEIGHTS_COLS {
                        let zero_point = weights
                            .zero_point
                            .get(output_channel)
                            .copied()
                            .unwrap_or(weights.zero_point[0]);
                        if view.mask[(filter_row, filter_cols)] {
                            accum[0][(filter_row, filter_cols)][output_channel] = accum[0]
                                [(filter_row, filter_cols)][output_channel]
                                .saturating_add(
                                    view.buffer[(filter_row, filter_cols)][output_channel]
                                        .saturating_sub(zero_point),
                                )
                                .saturating_mul(
                                    &output_grad[0][(output_row, output_col)][output_channel],
                                );
                        }
                    }
                }
            }
        }
    }
    array::from_fn(|batch| {
        let filters_scale = weights
            .scale
            .get(batch)
            .copied()
            .unwrap_or(weights.scale[0]);
        let scale = filters_scale * input.scale[0];
        SMatrix::from_fn(|i, j| {
            array::from_fn(|channel| {
                let tmp: f32 = T::to_superset(&accum[batch][(i, j)][channel]);
                T::from_superset(&(tmp * scale)).unwrap()
            })
        })
    })
}
pub fn grad_conv_2d_bias<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const FILTERS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    outputs: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    output_grad: &Buffer4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS>,
    activation: FusedActivation,
    bias_scale: [f32; FILTERS_QUANTS],
) -> SVector<f32, INPUT_CHANS> {
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    let mut accum: SVector<T, INPUT_CHANS> = SVector::zeros();
    for output_row in 0..INPUT_ROWS {
        for output_col in 0..WEIGHTS_COLS {
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
                accum[output_batch] = accum[output_batch]
                    .saturating_add(output_grad[0][(output_row, output_col)][output_batch]);
            }
        }
    }
    SMatrix::from_fn(|i, _| {
        let filters_scale = weights.scale.get(i).copied().unwrap_or(weights.scale[0]);
        let bias_scale_cur = bias_scale.get(i).copied().unwrap_or(bias_scale[0]);
        let scale = bias_scale_cur / (filters_scale * input.scale[0]).powi(2);
        let tmp: f32 = T::to_superset(&accum[i]);
        tmp * scale
    })
}
