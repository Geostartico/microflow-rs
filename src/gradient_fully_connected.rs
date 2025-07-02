use crate::{
    activation::FusedActivation,
    buffer::Buffer2D,
    quantize::{quantize, Trainable},
    tensor::Tensor2D,
    update_layer::update_weights_2D,
};
use nalgebra::{Matrix, SMatrix, SVector};

pub fn update_grad_fully_connected<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    output: Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights: &mut Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    activation: FusedActivation,
    output_grad: Buffer2D<T, INPUT_ROWS, WEIGHTS_COLS>,
    bias_scale: f32,
    learning_rate: f32,
) -> Buffer2D<T, INPUT_ROWS, INPUT_COLS> {
    let grad_weight =
        grad_fully_connected_weights(input, &output, weights, &activation, &output_grad);
    update_weights_2D(weights, grad_weight, learning_rate);
    grad_fully_connected_bias(
        input,
        &output,
        &weights,
        &activation,
        &output_grad,
        bias_scale,
    );
    grad_fully_connected_input(input, &output, weights, &activation, &output_grad)
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
    output_grad: &Buffer2D<T, INPUT_ROWS, WEIGHTS_COLS>,
) -> Buffer2D<T, INPUT_COLS, WEIGHTS_COLS> {
    //let scale = input.scale[0] * weights.scale[0]/weights.scale[0]powi(2);
    let scale = input.scale[0] / weights.scale[0];
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    let mut accum: Buffer2D<T, INPUT_COLS, WEIGHTS_COLS> = SMatrix::zeros();
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
                let tmp =
                    input.buffer[(output_row, weight_row)].saturating_sub(input.zero_point[0]);
                accum[(weight_row, output_col)] = accum[(weight_row, output_col)]
                    .saturating_add(tmp.saturating_mul(&output_grad[(output_row, output_col)]));
            }
        }
    }
    accum.map(|el| {
        let tmp: f32 = T::to_superset(&el);
        T::from_superset(&(tmp * scale).round()).unwrap()
    })
    //SMatrix::from_fn(|i,j|->T {
    //    T::from_superset(&(input.buffer
    //        .column(i)
    //        .map(|el| {
    //            let tmp : f32 = T::to_superset(& el.saturating_sub(input.zero_point[0]));
    //            input.scale[0] * weights.scale[0] * tmp
    //            })
    //        .component_mul(&output_grad.column(j).map(|el|T::to_superset(&el)))
    //        .sum()/(weights.scale[0].powi(2))).round()).unwrap()
    //
    //})
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
    output_grad: &Buffer2D<T, INPUT_ROWS, WEIGHTS_COLS>,
) -> Buffer2D<T, INPUT_ROWS, INPUT_COLS> {
    //let scale = input.scale[0] * weights.scale[0]/weights.scale[0]powi(2);
    let scale = weights.scale[0] / input.scale[0];
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    let mut accum: Buffer2D<T, INPUT_ROWS, INPUT_COLS> = SMatrix::zeros();
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
                let tmp =
                    weights.buffer[(weight_row, output_col)].saturating_sub(weights.zero_point[0]);
                accum[(output_row, weight_row)] = accum[(output_row, weight_row)]
                    .saturating_add(tmp.saturating_mul(&output_grad[(output_row, output_col)]));
            }
        }
    }
    accum.map(|el| {
        let tmp: f32 = T::to_superset(&el);
        T::from_superset(&(tmp * scale).round()).unwrap()
    })
    //SMatrix::from_fn(|i,j| {
    //    let back   = weights
    //        .buffer
    //        .row(j)
    //        .map(|el|el.saturating_sub(weights.zero_point[0]))
    //        .cast::<f32>()
    //        .scale(input.scale[0]*weights.scale[0])
    //        .component_mul(
    //            &output_grad
    //            .row(i)
    //            .map(|el|->f32 {T::to_superset(&el)})
    //            );
    //    let sum = back.sum();
    //    T::from_superset(&(sum/(input.scale[0].powi(2))).round()).unwrap()
    //})
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
    output_grad: &Buffer2D<T, INPUT_ROWS, WEIGHTS_COLS>,
    bias_scale: f32,
) -> SVector<f32, WEIGHTS_COLS> {
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    let mut accum: SVector<T, WEIGHTS_COLS> = SVector::zeros();
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
            accum[output_col] =
                accum[output_col].saturating_add(output_grad[(output_row, output_col)]);
        }
    }
    let scale = bias_scale / (weights.scale[0] * input.scale[0]).powi(2);
    accum.map(|el| {
        let tmp: f32 = T::to_superset(&el);
        tmp * scale
    })
}
