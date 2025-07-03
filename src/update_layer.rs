use crate::{
    buffer::{Buffer2D, Buffer4D},
    ops::softmax_borrow,
    quantize::{Quantized, Trainable},
    tensor::{Tensor2D, Tensor4D, TensorViewPadding},
};
use libm::logf;
use nalgebra::SMatrix;

pub fn update_weights_2D<T: Trainable, const ROWS: usize, const COLS: usize>(
    weights: &mut Tensor2D<T, ROWS, COLS, 1>,
    weights_gradient: Buffer2D<T, ROWS, COLS>,
    learning_rate: f32,
) {
    weights.buffer = SMatrix::from_fn(|i, j| {
        let tmp: f32 = T::to_superset(&weights_gradient[(i, j)]);
        weights.buffer[(i, j)]
            .saturating_sub(T::from_superset(&(learning_rate * tmp).round()).unwrap())
    });
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
    weights_gradient: Buffer4D<T, BATCHES, ROWS, COLS, CHANS>,
    learning_rate: f32,
) {
    weights.buffer = core::array::from_fn(|batch| {
        SMatrix::from_fn(|i, j| {
            core::array::from_fn(|channel| {
                let tmp: f32 = T::to_superset(&weights_gradient[batch][(i, j)][channel]);
                weights.buffer[batch][(i, j)][channel]
                    .saturating_sub(T::from_superset(&(learning_rate * tmp).round()).unwrap())
            })
        })
    })
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
    output_gt: Tensor2D<T, ROWS, COLS, 1>,
) -> Buffer2D<T, ROWS, COLS> {
    SMatrix::from_fn(|i, j| output_p.buffer[(i, j)].saturating_sub(output_gt.buffer[(i, j)]))
}
pub fn crossentropy_grad<T: Trainable, const ROWS: usize, const COLS: usize>(
    input: &Tensor2D<T, ROWS, COLS, 1>,
    output_scale: f32,
    output_zero_point: T,
    label: &Tensor2D<T, ROWS, COLS, 1>,
) -> Buffer2D<T, ROWS, COLS> {
    let softm = softmax_borrow(&input, [output_scale], [output_zero_point]);
    SMatrix::from_fn(|i, j| {
        let diff: f32 = T::to_superset(&softm.buffer[(i, j)].saturating_sub(label.buffer[(i, j)]));
        T::from_superset(&(output_scale * diff / (input.scale[0].powi(2)))).unwrap()
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
