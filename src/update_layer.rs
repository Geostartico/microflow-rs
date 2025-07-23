use crate::{
    buffer::{Buffer2D, Buffer4D},
    ops::softmax_borrow,
    quantize::{Quantized, Trainable},
    tensor::{Tensor2D, Tensor4D, TensorViewPadding},
};
use libm::logf;
use nalgebra::SMatrix;
use simba::scalar::SupersetOf;

pub fn update_weights_2D<T: Trainable, const ROWS: usize, const COLS: usize>(
    weights: &mut Tensor2D<T, ROWS, COLS, 1>,
    weights_gradient: &Buffer2D<i32, ROWS, COLS>,
    batch_size: usize,
    learning_rate: f32,
) {
    weights.buffer = SMatrix::from_fn(|i, j| {
        let tmp: f32 = weights_gradient[(i, j)] as f32;
        weights.buffer[(i, j)].saturating_sub(
            T::from_superset(&(learning_rate * tmp / batch_size as f32).round()).unwrap(),
        )
    });
}
pub fn update_weights_2D_float<const ROWS: usize, const COLS: usize>(
    weights: &mut Buffer2D<f32, ROWS, COLS>,
    weights_gradient: &Buffer2D<f32, ROWS, COLS>,
    batch_size: usize,
    learning_rate: f32,
) {
    for row in 0..ROWS {
        for col in 0..COLS {
            weights[(row, col)] -= learning_rate * weights_gradient[(row, col)] / batch_size as f32
        }
    }
}

pub fn update_weights_4D<
    T: Trainable,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
    const QUANTS: usize,
>(
    weights: &mut Tensor4D<T, BATCHES, ROWS, COLS, CHANS, QUANTS>,
    weights_gradient: &Buffer4D<i32, BATCHES, ROWS, COLS, CHANS>,
    batch_size: usize,
    learning_rate: f32,
) {
    weights.buffer = core::array::from_fn(|batch| {
        SMatrix::from_fn(|i, j| {
            core::array::from_fn(|channel| {
                let tmp: f32 = weights_gradient[batch][(i, j)][channel] as f32;
                weights.buffer[batch][(i, j)][channel].saturating_sub(
                    T::from_superset(&(learning_rate * tmp / batch_size as f32).round()).unwrap(),
                )
            })
        })
    })
}
pub fn accumulate_gradient_2D<T: Trainable, const ROWS: usize, const COLS: usize>(
    current_gradient: &Buffer2D<T, ROWS, COLS>,
    weights_gradient: &mut Buffer2D<i32, ROWS, COLS>,
) {
    for row in 0..ROWS {
        for col in 0..COLS {
            let tmp: i32 = current_gradient[(row, col)].to_superset();
            weights_gradient[(row, col)] += tmp;
        }
    }
}

pub fn accumulate_gradient_4D<
    T: Trainable,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
>(
    current_gradient: &Buffer4D<T, BATCHES, ROWS, COLS, CHANS>,
    weights_gradient: &mut Buffer4D<i32, BATCHES, ROWS, COLS, CHANS>,
) {
    for batch in 0..BATCHES {
        for row in 0..ROWS {
            for col in 0..COLS {
                for channel in 0..CHANS {
                    let tmp: i32 = current_gradient[batch][(row, col)][channel].to_superset();
                    weights_gradient[batch][(row, col)][channel] =
                        tmp.saturating_add(weights_gradient[batch][(row, col)][channel]);
                }
            }
        }
    }
}

pub fn mse_loss<T: Quantized, const ROWS: usize, const COLS: usize>(
    output_p: &Tensor2D<T, ROWS, COLS, 1>,
    output_gt: &Tensor2D<T, ROWS, COLS, 1>,
) -> f32 {
    let difference: Buffer2D<f32, ROWS, COLS> = SMatrix::from_fn(|i, j| {
        let casted_p: f32 = T::to_superset(&output_p.buffer[(i, j)]);
        let casted_gt: f32 = T::to_superset(&output_gt.buffer[(i, j)]);
        output_p.scale[0] * (casted_p - casted_gt)
    });
    0.5f32 * difference.component_mul(&difference).sum()
}

pub fn mse_grad<T: Trainable, const ROWS: usize, const COLS: usize>(
    output_p: &Tensor2D<T, ROWS, COLS, 1>,
    output_gt: &Tensor2D<T, ROWS, COLS, 1>,
) -> Buffer2D<i32, ROWS, COLS> {
    SMatrix::from_fn(|i, j| {
        i32::from_subset(&output_p.buffer[(i, j)]) - i32::from_subset(&output_gt.buffer[(i, j)])
    })
}
pub fn crossentropy_grad<T: Trainable, const ROWS: usize, const COLS: usize>(
    input: &Tensor2D<T, ROWS, COLS, 1>,
    output_scale: f32,
    output_zero_point: T,
    label: &Tensor2D<T, ROWS, COLS, 1>,
) -> Buffer2D<i32, ROWS, COLS> {
    let softm = softmax_borrow(&input, [output_scale], [output_zero_point]);
    SMatrix::from_fn(|i, j| {
        let tmp1: i32 = T::to_superset(&softm.buffer[(i, j)]);
        let tmp2: i32 = label.buffer[(i, j)].to_superset();
        let diff: i32 = tmp1 - tmp2;
        // T::from_superset(&(output_scale * diff / (input.scale[0].powi(2)))).unwrap()
        diff
    })
}
pub fn cross_entropy_loss<T: Trainable, const ROWS: usize, const COLS: usize>(
    input: &Tensor2D<T, ROWS, COLS, 1>,
    output_scale: f32,
    output_zero_point: T,
    label: &Tensor2D<T, ROWS, COLS, 1>,
) -> f32 {
    let softm = softmax_borrow(&input, [output_scale], [output_zero_point]);
    let label = label.dequantize();
    label
        .component_mul(&softm.dequantize().map(|el| logf(el)))
        .sum()
}

pub fn get_input_index(
    view_rows: usize,
    view_cols: usize,
    focus: (usize, usize),
    padding: TensorViewPadding,
    strides: (usize, usize),
) -> (i32, i32) {
    match padding {
        TensorViewPadding::Same => {
            let shift = ((view_rows - 1) / 2, (view_cols - 1) / 2);
            (
                (strides.0 * focus.0) as i32 - (shift.0) as i32,
                (strides.1 * focus.1) as i32 - (shift.1) as i32,
            )
        }
        TensorViewPadding::Valid => ((strides.0 * focus.0) as i32, (strides.1 * focus.1) as i32),
    }
}
