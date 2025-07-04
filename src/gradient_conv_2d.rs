use crate::{
    activation::FusedActivation,
    buffer::{Buffer2D, Buffer4D},
    quantize::{quantize, Trainable},
    tensor::{Tensor4D, TensorView, TensorViewPadding},
    update_layer::{get_input_index, update_weights_4D},
};
use core::array;
use nalgebra::{SMatrix, SVector};

pub fn update_grad_conv_2d<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const FILTER_QUANTS: usize,
    const FILTER_NUM: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &mut Tensor4D<T, FILTER_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTER_QUANTS>,
    constants: &mut (
        Buffer2D<f32, FILTER_NUM, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    outputs: Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTER_NUM, 1>,
    output_grad: Buffer4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTER_NUM>,
    activation: FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
    bias_scale: [f32; FILTER_QUANTS],
    learning_rate: f32,
) -> Buffer4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let grad_weight = grad_conv_2d_weights(
        input,
        weights,
        &outputs,
        &output_grad,
        &activation,
        strides,
        padding,
    );
    update_weights_4D(weights, grad_weight, learning_rate);
    let grad_bias = grad_conv_2d_bias(
        input,
        weights,
        &outputs,
        &output_grad,
        &activation,
        bias_scale,
    );
    update_bias_conv2d(constants, grad_bias, learning_rate);
    grad_conv_2d_inputs(
        input,
        weights,
        &outputs,
        &output_grad,
        &activation,
        strides,
        padding,
    )
}
pub fn update_bias_conv2d<const FILTER_QUANTS: usize, const FILTER_NUM: usize>(
    constants: &mut (
        Buffer2D<f32, FILTER_NUM, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    bias_gradient: Buffer2D<f32, FILTER_NUM, 1>,
    learning_rate: f32,
) {
    constants.0 =
        SMatrix::from_fn(|i, j| constants.0[(i, j)] - learning_rate * bias_gradient[(i, j)]);
}
pub fn grad_conv_2d_inputs<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const FILTERS_NUM: usize,
    const FILTERS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM, 1>,
    output_grad: &Buffer4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| array::from_fn(|_| T::zero())));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            for output_batch in 0..FILTERS_NUM {
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
                    for filter_col in 0..WEIGHTS_COLS {
                        if (coord.1 + filter_col as i32) < 0 {
                            continue;
                        }
                        for filter_chans in 0..INPUT_CHANS {
                            let cur_coord = (
                                (coord.0 + filter_row as i32) as usize,
                                (coord.1 + filter_col as i32) as usize,
                            );
                            let filters_zero_point = &weights
                                .zero_point
                                .get(output_batch)
                                .copied()
                                .unwrap_or(weights.zero_point[0]);
                            accum[output_batch][cur_coord][filter_chans] = accum[output_batch]
                                [cur_coord][filter_chans]
                                .saturating_add(
                                    input.buffer[0][cur_coord][filter_chans]
                                        .saturating_sub(*filters_zero_point),
                                )
                                .saturating_mul(
                                    &output_grad[0][(output_row, output_col)][output_batch],
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
        let scale = filters_scale / (input.scale[0] * outputs.scale[0]);
        SMatrix::from_fn(|i, j| {
            array::from_fn(|channel| {
                let tmp: f32 = T::to_superset(&accum[batch][(i, j)][channel]);
                T::from_superset(&(tmp * scale)).unwrap()
            })
        })
    })
}

pub fn grad_conv_2d_weights<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const FILTERS_NUM: usize,
    const FILTERS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM, 1>,
    output_grad: &Buffer4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<T, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<T, FILTERS_NUM, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| [T::zero(); INPUT_CHANS]));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            for output_batch in 0..FILTERS_NUM {
                let val = outputs.buffer[0][(output_row, output_col)][output_batch]
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
                let filters_zero_point = &weights
                    .zero_point
                    .get(output_batch)
                    .copied()
                    .unwrap_or(weights.zero_point[0]);
                for filter_row in 0..WEIGHTS_ROWS {
                    for filter_cols in 0..WEIGHTS_COLS {
                        for filter_chans in 0..INPUT_CHANS {
                            if view.mask[(filter_row, filter_cols)] {
                                accum[output_batch][(filter_row, filter_cols)][filter_chans] =
                                    accum[output_batch][(filter_row, filter_cols)][filter_chans]
                                        .saturating_add(
                                            view.buffer[(filter_row, filter_cols)][filter_chans]
                                                .saturating_sub(*filters_zero_point),
                                        )
                                        .saturating_mul(
                                            &output_grad[0][(output_row, output_col)][output_batch],
                                        );
                            }
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
        let scale = input.scale[0] / (filters_scale * outputs.scale[0]);
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
    const FILTERS_NUM: usize,
    const FILTERS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM, 1>,
    output_grad: &Buffer4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM>,
    activation: &FusedActivation,
    bias_scale: [f32; FILTERS_QUANTS],
) -> SVector<f32, FILTERS_NUM> {
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    let mut accum: SVector<T, FILTERS_NUM> = SVector::zeros();
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            for output_batch in 0..FILTERS_NUM {
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
        //let bias_scale_cur = bias_scale.get(i).copied().unwrap_or(bias_scale[0]);
        //let scale = bias_scale_cur / (filters_scale * input.scale[0]).powi(2);
        let scale = 1f32 / (filters_scale * input.scale[0]).powi(2);
        let tmp: f32 = T::to_superset(&accum[i]);
        tmp * scale
    })
}
