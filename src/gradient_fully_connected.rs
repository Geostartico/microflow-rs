use crate::{
    activation::FusedActivation,
    buffer::Buffer2D,
    quantize::{quantize, Trainable},
    tensor::Tensor2D,
    update_layer::{accumulate_gradient_2D, update_weights_2D},
};
use nalgebra::{SMatrix, SVector};
use simba::scalar::{SubsetOf, SupersetOf};

pub fn update_grad_fully_connected<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    output: &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights: &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    weights_gradient: &mut Buffer2D<i32, INPUT_COLS, WEIGHTS_COLS>,
    constants: &(
        Buffer2D<f32, WEIGHTS_COLS, 1>,
        f32,
        Buffer2D<i32, 1, WEIGHTS_COLS>,
        i32,
    ),
    constants_gradient: &mut (
        Buffer2D<f32, WEIGHTS_COLS, 1>,
        f32,
        Buffer2D<i32, 1, WEIGHTS_COLS>,
        i32,
    ),
    activation: FusedActivation,
    output_grad: Buffer2D<i32, INPUT_ROWS, WEIGHTS_COLS>,
    bias_scale: f32,
    learning_rate: f32,
) -> Buffer2D<i32, INPUT_ROWS, INPUT_COLS> {
    let grad_weight =
        grad_fully_connected_weights(input, &output, weights, &activation, &output_grad);
    accumulate_gradient_2D(&grad_weight, weights_gradient);
    let grad_bias = grad_fully_connected_bias(
        input,
        &output,
        &weights,
        &activation,
        &output_grad,
        bias_scale,
    );
    update_bias_fully_connected(constants_gradient, grad_bias);
    grad_fully_connected_input(input, &output, weights, &activation, &output_grad)
}
pub fn update_bias_fully_connected<const WEIGHTS_COLS: usize>(
    constants: &mut (
        Buffer2D<f32, WEIGHTS_COLS, 1>,
        f32,
        Buffer2D<i32, 1, WEIGHTS_COLS>,
        i32,
    ),
    bias_gradient: Buffer2D<f32, WEIGHTS_COLS, 1>,
) {
    constants.0 = SMatrix::from_fn(|i, j| constants.0[(i, j)] + bias_gradient[(i, j)]);
}
pub fn grad_fully_connected_weights<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    output: &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights: &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    activation: &FusedActivation,
    output_grad: &Buffer2D<i32, INPUT_ROWS, WEIGHTS_COLS>,
) -> Buffer2D<i32, INPUT_COLS, WEIGHTS_COLS> {
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    let mut accum: Buffer2D<i32, INPUT_COLS, WEIGHTS_COLS> = SMatrix::zeros();
    let mut normalization_factor: Buffer2D<i32, INPUT_COLS, WEIGHTS_COLS> = SMatrix::zeros();
    for output_row in 0..INPUT_ROWS {
        for output_col in 0..WEIGHTS_COLS {
            let val = output.buffer[(output_row, output_col)].saturating_sub(output.zero_point[0]);
            if !(match activation {
                FusedActivation::Relu => val > T::zero(),
                FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                _ => true,
            }) {
                continue;
            }
            for weight_row in 0..INPUT_COLS {
                let tmp = i32::from_subset(&input.buffer[(output_row, weight_row)])
                    - i32::from_subset(&input.zero_point[0]);
                accum[(weight_row, output_col)] +=
                    tmp * i32::from_subset(&output_grad[(output_row, output_col)]);
                normalization_factor[(weight_row, output_col)] +=
                    i32::from_subset(&output_grad[(output_row, output_col)]).abs();
            }
        }
    }
    SMatrix::from_fn(|i, j| {
        if normalization_factor[(i, j)] != 0 {
            accum[(i, j)] / normalization_factor[(i, j)]
        } else {
            0
        }
    })
}

pub fn grad_fully_connected_input<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    output: &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights: &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    activation: &FusedActivation,
    output_grad: &Buffer2D<i32, INPUT_ROWS, WEIGHTS_COLS>,
) -> Buffer2D<i32, INPUT_ROWS, INPUT_COLS> {
    let mut accum: Buffer2D<i32, INPUT_ROWS, INPUT_COLS> = SMatrix::zeros();
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    let mut normalization_factor: Buffer2D<i32, INPUT_ROWS, INPUT_COLS> = SMatrix::zeros();
    for output_row in 0..INPUT_ROWS {
        for output_col in 0..WEIGHTS_COLS {
            let val = output.buffer[(output_row, output_col)];
            if !(match activation {
                FusedActivation::Relu => val > T::zero(),
                FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                _ => true,
            }) {
                continue;
            }
            for weight_row in 0..INPUT_COLS {
                let tmp = i32::from_subset(&weights.buffer[(weight_row, output_col)])
                    - i32::from_subset(&weights.zero_point[0]);
                accum[(output_row, weight_row)] +=
                    tmp * i32::from_subset(&output_grad[(output_row, output_col)]);
                normalization_factor[(output_row, weight_row)] +=
                    i32::from_subset(&output_grad[(output_row, output_col)]).abs();
            }
        }
    }
    SMatrix::from_fn(|i, j| {
        if normalization_factor[(i, j)] != 0 {
            accum[(i, j)] / normalization_factor[(i, j)]
        } else {
            0
        }
    })
}
pub fn grad_fully_connected_bias<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    output: &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights: &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    activation: &FusedActivation,
    output_grad: &Buffer2D<i32, INPUT_ROWS, WEIGHTS_COLS>,
    bias_scale: f32,
) -> SVector<f32, WEIGHTS_COLS> {
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    let mut accum: SVector<i32, WEIGHTS_COLS> = SVector::zeros();
    for output_row in 0..INPUT_ROWS {
        for output_col in 0..WEIGHTS_COLS {
            let val = output.buffer[(output_row, output_col)].saturating_sub(output.zero_point[0]);
            if !(match activation {
                FusedActivation::Relu => val > T::zero(),
                FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                _ => true,
            }) {
                continue;
            }
            accum[output_col] += output_grad[(output_row, output_col)];
        }
    }
    //let scale = bias_scale / (weights.scale[0] * input.scale[0]).powi(2);
    //let scale = 1f32 / (weights.scale[0] * input.scale[0]).powi(2);
    accum.map(|el| {
        let tmp: f32 = i32::to_superset(&el);
        tmp // * scale
    })
}
