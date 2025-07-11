use crate::{
    activation::FusedActivation,
    buffer::{Buffer2D, Buffer4D},
    quantize::{quantize, Trainable},
    tensor::{Tensor4D, TensorView, TensorViewPadding},
    update_layer::{get_input_index, update_weights_4D},
};
use core::{array, ops::Mul};
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
    output_grad: Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTER_NUM>,
    activation: FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
    bias_scale: [f32; FILTER_QUANTS],
    learning_rate: f32,
) -> (
    Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS>,
    Buffer4D<T, FILTER_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS>,
) {
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
    (
        grad_conv_2d_inputs(
            input,
            weights,
            &outputs,
            &output_grad,
            &activation,
            strides,
            padding,
        ),
        grad_weight,
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
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| array::from_fn(|_| 0i32)));
    let mut normalization_param: Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| array::from_fn(|_| 0i32)));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_batch in 0..FILTERS_NUM {
        for output_row in 0..OUTPUT_ROWS {
            for output_col in 0..OUTPUT_COLS {
                let coord = get_input_index(
                    WEIGHTS_ROWS,
                    WEIGHTS_COLS,
                    (output_row, output_col),
                    padding,
                    strides,
                );
                let val = outputs.buffer[0][(output_row, output_col)][output_batch]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                for filter_row in 0..WEIGHTS_ROWS {
                    if (coord.0 + filter_row as i32) < 0
                        || (coord.0 + filter_row as i32) >= INPUT_ROWS as i32
                    {
                        continue;
                    }
                    for filter_col in 0..WEIGHTS_COLS {
                        if (coord.1 + filter_col as i32) < 0
                            || (coord.1 + filter_col as i32) >= INPUT_COLS as i32
                        {
                            continue;
                        }
                        for filter_chans in 0..INPUT_CHANS {
                            let cur_coord = (
                                (coord.0 + filter_row as i32) as usize,
                                (coord.1 + filter_col as i32) as usize,
                            );
                            let filters_zero_point: i32 = weights
                                .zero_point
                                .get(output_batch)
                                .copied()
                                .unwrap_or(weights.zero_point[0])
                                .to_superset();

                            let tmp: i32 = weights.buffer[output_batch][(filter_row, filter_col)]
                                [filter_chans]
                                .to_superset();
                            accum[0][cur_coord][filter_chans] += (tmp - filters_zero_point)
                                * output_grad[0][(output_row, output_col)][output_batch];
                            normalization_param[0][cur_coord][filter_chans] +=
                                output_grad[0][(output_row, output_col)][output_batch].abs();
                        }
                    }
                }
            }
        }
    }
    [SMatrix::from_fn(|i, j| {
        array::from_fn(|k| {
            (accum[0][(i, j)][k] as f32 / normalization_param[0][(i, j)][k] as f32).round() as i32
        })
    })]
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
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<T, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> {
    let mut normalization_param: Buffer4D<
        i32,
        FILTERS_NUM,
        WEIGHTS_ROWS,
        WEIGHTS_COLS,
        INPUT_CHANS,
    > = array::from_fn(|_| SMatrix::from_fn(|_, _| [0i32; INPUT_CHANS]));
    let mut accum: Buffer4D<i32, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| [0i32; INPUT_CHANS]));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            let view: TensorView<T, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> =
                input.view((output_row, output_col), 0, padding, strides);
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
                let input_zero_point: i32 = input.zero_point[0].to_superset();
                for filter_row in 0..WEIGHTS_ROWS {
                    for filter_cols in 0..WEIGHTS_COLS {
                        for filter_chans in 0..INPUT_CHANS {
                            if view.mask[(filter_row, filter_cols)] {
                                let tmp: i32 = view.buffer[(filter_row, filter_cols)][filter_chans]
                                    .to_superset();
                                accum[output_batch][(filter_row, filter_cols)][filter_chans] +=
                                    (tmp - input_zero_point).mul(
                                        output_grad[0][(output_row, output_col)][output_batch],
                                    );
                                normalization_param[output_batch][(filter_row, filter_cols)]
                                    [filter_chans] +=
                                    output_grad[0][(output_row, output_col)][output_batch].abs();
                            }
                        }
                    }
                }
            }
        }
    }
    array::from_fn(|batch| {
        SMatrix::from_fn(|i, j| {
            array::from_fn(|k| {
                T::from_superset_unchecked(
                    &(accum[batch][(i, j)][k] as f32
                        / normalization_param[batch][(i, j)][k] as f32)
                        .round(),
                )
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
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM>,
    activation: &FusedActivation,
    bias_scale: [f32; FILTERS_QUANTS],
) -> SVector<f32, FILTERS_NUM> {
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    let mut accum: SVector<i32, FILTERS_NUM> = SVector::zeros();
    let normalization_param = output_grad.iter().fold(0f32, |acc, val| {
        acc + val.iter().fold(0f32, |acc1, val1| {
            acc1 + val1.iter().fold(0f32, |acc2, val2| acc2 + *val2 as f32)
        })
    });
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
                accum[output_batch] =
                    accum[output_batch] + output_grad[0][(output_row, output_col)][output_batch];
            }
        }
    }
    SMatrix::from_fn(|i, _| {
        let filters_scale = weights.scale.get(i).copied().unwrap_or(weights.scale[0]);
        // let scale = 1f32 / (filters_scale * input.scale[0]).powi(2);
        let tmp: f32 = accum[i] as f32;
        // tmp * scale
        tmp / normalization_param
    })
}
