use core::{array, iter::Filter};

use libm::{expf, log, logf};
use nalgebra::SMatrix;
use simba::scalar::SupersetOf;
use crate::{
    activation::{self, FusedActivation},
    buffer::{Buffer2D, Buffer4D},
    ops::{softmax, softmax_borrow},
    quantize::{dequantize, quantize, Quantized, Trainable},
    tensor::{Tensor2D, Tensor4D, TensorView, TensorViewPadding}
};
fn update_weights_2D<
    T : Trainable,
    const ROWS : usize,
    const COLS : usize,
>(
    weights : &mut Tensor2D<T,ROWS,COLS, 1>,
    weights_gradient : Tensor2D<T,ROWS,COLS, 1>,
    learning_rate : f32,
){
    weights.buffer = SMatrix::from_fn(|i, j |{
        let shifted : f32 = T::to_superset(&weights_gradient.buffer[(i,j)].saturating_sub(weights_gradient.zero_point[0]));
        weights.buffer[(i,j)].saturating_sub(
            T::from_superset(&(learning_rate
              * weights_gradient.scale[0]
              * shifted).round()).unwrap())
    });
}

fn update_weights_4D<
    T : Trainable,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
    const QUANTS: usize,
>(
    weights : &mut Tensor4D<T,BATCHES, ROWS,COLS,CHANS, QUANTS>,
    weights_gradient : Tensor4D<T,BATCHES, ROWS,COLS,CHANS, QUANTS>,
    learning_rate : f32,
){
    weights.buffer = core::array::from_fn(|batch|{
        let weights_zero_point = &weights
            .zero_point
            .get(batch)
            .copied()
            .unwrap_or(weights.zero_point[0]);
        let weights_gradient_zero_point = &weights_gradient
            .zero_point
            .get(batch)
            .copied()
            .unwrap_or(weights.zero_point[0]);
        let weights_scale = &weights
            .scale
            .get(batch)
            .copied()
            .unwrap_or(weights.scale[0]);
        let weights_gradient_scale = &weights
            .scale
            .get(batch)
            .copied()
            .unwrap_or(weights.scale[0]);
        SMatrix::from_fn(|i, j |{
            core::array::from_fn(|channel|{
                let shifted : f32 = T::to_superset( &weights_gradient.buffer[batch][(i,j)][channel].saturating_sub(*weights_gradient_zero_point));
                            weights.buffer[batch][(i,j)][channel]
                                .saturating_sub(T::from_superset(&(learning_rate
                            * weights_gradient_scale
                            * shifted).round()).unwrap())
            })
        })
    })
}

fn mse_loss<
    T : Quantized,
    const ROWS : usize,
    const COLS : usize,
>(
    output_p: &mut Tensor2D<T,ROWS,COLS, 1>,
    output_gt: Tensor2D<T,ROWS,COLS, 1>,
) -> f32{
    let difference : Buffer2D<f32, ROWS, COLS> = SMatrix::from_fn(|i,j|{
        let casted_p : f32= T::to_superset(&output_p.buffer[(i,j)]);
        let casted_gt : f32 = T::to_superset(&output_gt.buffer[(i,j)]);
        output_p.scale[0] * (casted_p - casted_gt)
    });
    0.5f32 * difference
        .component_mul(&difference)
        .sum()
        
}

fn mse_grad<
    T : Trainable,
    const ROWS : usize,
    const COLS : usize,
>(
    output_p: &mut Tensor2D<T,ROWS,COLS, 1>,
    output_gt: Tensor2D<T,ROWS,COLS, 1>,
) -> Buffer2D<T,ROWS,COLS>{
    SMatrix::from_fn(|i,j|{
        output_p.buffer[(i,j)].saturating_sub(output_gt.buffer[(i,j)])
    })
}
pub fn grad_cross_entropy_loss<
T: Trainable,
const ROWS: usize,
const COLS: usize,
>(
     input : &Tensor2D<T, ROWS,COLS,1>,
     output_scale : f32,
     output_zero_point : T,
     label : &Tensor2D<T, ROWS,COLS,1>,
    )->Buffer2D<T,ROWS,COLS>{
    let softm = softmax_borrow(&input, [output_scale], [output_zero_point]);
    SMatrix::from_fn(|i,j|{
        let diff : f32 = T::to_superset(&softm.buffer[(i,j)].saturating_sub(label.buffer[(i,j)]));
        T::from_superset(&(output_scale*diff/(input.scale[0].powi(2)))).unwrap()
    })

}
pub fn cross_entropy_loss<
T: Trainable,
const ROWS: usize,
const COLS: usize,
>(
     input : &Tensor2D<T, ROWS,COLS,1>,
     output_scale : f32,
     output_zero_point : T,
     label : &Tensor2D<T, ROWS,COLS,1>,
    )->f32{
    let softm = softmax_borrow(&input, [output_scale], [output_zero_point]);
    let label = label.dequantize();
    label.component_mul(&softm
        .dequantize()
        .map(|el| logf(el))).sum()
}
pub fn grad_fully_connected_weights<
T: Trainable,
const INPUT_ROWS: usize,
const INPUT_COLS: usize,
const WEIGHTS_COLS: usize,
>(
    input : &Tensor2D<T, INPUT_ROWS, INPUT_COLS,1>,
    output : &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights : &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS,1>,
    activation : FusedActivation,
    output_grad : &Buffer2D<T, INPUT_ROWS, WEIGHTS_COLS>
)->Buffer2D<T,INPUT_COLS,WEIGHTS_COLS>{
    //let scale = input.scale[0] * weights.scale[0]/weights.scale[0]powi(2);
    let scale = input.scale[0] / weights.scale[0];
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    let mut accum : Buffer2D<T,INPUT_COLS,WEIGHTS_COLS> = SMatrix::zeros();
    for output_row in 0..INPUT_ROWS{
        for output_col in 0..WEIGHTS_COLS{
            let val = output.buffer[(output_row,output_col)];
            if !(match activation {
                FusedActivation::Relu => {val > T::zero()},
                FusedActivation::Relu6 => {val > T::zero() && val < quantized_6},
                _ => true
            }){
                continue;
            }
            for weight_row in 0..INPUT_COLS{
                let tmp = input.buffer[(output_row, weight_row)].saturating_sub(input.zero_point[0]);
                accum[(weight_row, output_col)] = accum[(weight_row, output_col)].saturating_add(tmp.saturating_mul(&output_grad[(output_row, output_col)]));
            }
        }
    }
    accum
        .map(|el| {
            let tmp : f32 = T::to_superset(&el);
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
const WEIGHTS_COLS: usize
>(
    input : &Tensor2D<T,INPUT_ROWS,INPUT_COLS,1>,
    output : &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights : &Tensor2D<T,INPUT_COLS,WEIGHTS_COLS,1>,
    activation : FusedActivation,
    output_grad : &Buffer2D<T,INPUT_ROWS,WEIGHTS_COLS>

)->Buffer2D<T,INPUT_ROWS,INPUT_COLS>{
    //let scale = input.scale[0] * weights.scale[0]/weights.scale[0]powi(2);
    let scale = weights.scale[0] / input.scale[0];
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    let mut accum : Buffer2D<T,INPUT_ROWS,INPUT_COLS> = SMatrix::zeros();
    for output_row in 0..INPUT_ROWS{
        for output_col in 0..WEIGHTS_COLS{
            let val = output.buffer[(output_row,output_col)];
            if !(match activation {
                FusedActivation::Relu => {val > T::zero()},
                FusedActivation::Relu6 => {val > T::zero() && val < quantized_6},
                _ => true
            }){
                continue;
            }
            for weight_row in 0..INPUT_COLS{
                let tmp = weights.buffer[(weight_row, output_col)].saturating_sub(weights.zero_point[0]);
                accum[(output_row, weight_row)] = accum[(output_row, weight_row)].saturating_add(tmp.saturating_mul(&output_grad[(output_row, output_col)]));
            }
        }
    }
    accum
        .map(|el| {
            let tmp : f32 = T::to_superset(&el);
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
>(
    input : &Tensor4D<T,1,INPUT_ROWS,INPUT_COLS,INPUT_CHANS, 1>,
    weights : &Tensor4D<T,FILTERS_NUM,WEIGHTS_ROWS,WEIGHTS_COLS,INPUT_CHANS, 1>,
    outputs : &Tensor4D<T,1,INPUT_ROWS,INPUT_COLS,FILTERS_NUM, 1>,
    output_grad : &Buffer4D<T,1,INPUT_ROWS,INPUT_COLS,FILTERS_NUM>,
    activation : FusedActivation,
    strides: (usize, usize),
    padding : TensorViewPadding,

)->Buffer4D<T,FILTERS_NUM,WEIGHTS_ROWS,WEIGHTS_COLS,INPUT_CHANS>{
    let mut accum : Buffer4D<T,FILTERS_NUM,INPUT_ROWS,INPUT_COLS,INPUT_CHANS> =
        array::from_fn(|_|{SMatrix::from_fn(|_,_|[T::zero(); INPUT_CHANS])});
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in (0..INPUT_ROWS).step_by(strides.0){
        for output_col  in (0..INPUT_COLS).step_by(strides.1){
            for output_batch  in (0..FILTERS_NUM).step_by(strides.1){
                let val = outputs.buffer[0][(output_row,output_col)][output_batch];
                if !(match activation {
                    FusedActivation::Relu => {val > T::zero()},
                    FusedActivation::Relu6 => {val > T::zero() && val < quantized_6},
                    _ => true
                }){
                    continue;
                }
                let view : TensorView<T, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> = input.view((output_row,output_col),
                0,
                padding,
                strides);
                for filter_row in 0..WEIGHTS_ROWS{
                    for filter_cols in 0..WEIGHTS_COLS{
                        for filter_chans in 0..INPUT_CHANS{
                            if view.mask[(filter_row,filter_cols)]{
                                accum[output_batch][(filter_row,filter_cols)][filter_chans] =
                                    accum[output_batch][(filter_row,filter_cols)][filter_chans]
                                        .saturating_add(view.buffer[(filter_row,filter_cols)][filter_chans]
                                        .saturating_sub(input.zero_point[0]))
                                        .saturating_mul(& output_grad[0][(output_row,output_col)][output_batch])
                                    ;
                            }
                        }
                    }
                }
            }
        }
    }
    array::from_fn(|batch|{
        let filters_scale = 
            weights
            .scale
            .get(batch)
            .copied()
            .unwrap_or(weights.scale[0]);
        let scale = filters_scale * input.scale[0];
        SMatrix::from_fn(|i,j|{
            array::from_fn(|channel|{
                let tmp : f32 = T::to_superset(&accum[batch][(i,j)][channel]);
                T::from_superset(& (tmp * scale)).unwrap()
            })
        })
    }
    )
        
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
>(
    input : &Tensor4D<T,1,INPUT_ROWS,INPUT_COLS,INPUT_CHANS, 1>,
    weights : &Tensor4D<T,FILTERS_NUM,WEIGHTS_ROWS,WEIGHTS_COLS,INPUT_CHANS, 1>,
    outputs : &Tensor4D<T,1,INPUT_ROWS,INPUT_COLS,FILTERS_NUM, 1>,
    output_grad : &Buffer4D<T,1,INPUT_ROWS,INPUT_COLS,FILTERS_NUM>,
    activation : FusedActivation,
    strides: (usize, usize),
    padding : TensorViewPadding,

)->Buffer4D<T,FILTERS_NUM,WEIGHTS_ROWS,WEIGHTS_COLS,INPUT_CHANS>{
    let mut accum : Buffer4D<T,1,INPUT_ROWS,INPUT_COLS,INPUT_CHANS> =
        array::from_fn(|_|{SMatrix::from_fn(|_,_|[T::zero(); INPUT_CHANS])});
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in (0..INPUT_ROWS).step_by(strides.0){
        for output_col  in (0..INPUT_COLS).step_by(strides.1){
            for output_batch  in (0..FILTERS_NUM).step_by(strides.1){
                let val = outputs.buffer[0][(output_row,output_col)][output_batch];
                if !(match activation {
                    FusedActivation::Relu => {val > T::zero()},
                    FusedActivation::Relu6 => {val > T::zero() && val < quantized_6},
                    _ => true
                }){
                    continue;
                }
                let coord = get_input_index(WEIGHTS_ROWS, WEIGHTS_COLS, (output_row,output_col), padding, strides);
                for filter_row in 0..WEIGHTS_ROWS{
                    if (coord.0 + filter_row as i32) < 0 {
                        continue;
                    }
                    for filter_col in 0..WEIGHTS_ROWS{
                        if (coord.1 + filter_col as i32) < 0{
                            continue;
                        }
                        for filter_chans in 0..INPUT_CHANS{
                            let cur_coord = ((coord.0 + filter_row as i32) as usize, (coord.1 + filter_col as i32) as usize);
                                accum[output_batch][cur_coord][filter_chans] =
                                    accum[output_batch][cur_coord][filter_chans]
                                        .saturating_add(input.buffer[0][cur_coord][filter_chans]
                                        .saturating_sub(weights.zero_point[0]))
                                        .saturating_mul(& output_grad[0][(output_row,output_col)][output_batch])
                                    ;
                            }
                        }
                    }
                }
            }
        }
    array::from_fn(|batch|{
        let filters_scale = 
            weights
            .scale
            .get(batch)
            .copied()
            .unwrap_or(weights.scale[0]);
        let scale = filters_scale * input.scale[0];
        SMatrix::from_fn(|i,j|{
            array::from_fn(|channel|{
                let tmp : f32 = T::to_superset(&accum[batch][(i,j)][channel]);
                T::from_superset(& (tmp * scale)).unwrap()
            })
        })
    }
    )
        
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
>(
    input : &Tensor4D<T,1,INPUT_ROWS,INPUT_COLS,INPUT_CHANS, 1>,
    weights : &Tensor4D<T,1,WEIGHTS_ROWS,WEIGHTS_COLS,INPUT_CHANS, 1>,
    outputs : &Tensor4D<T,1,INPUT_ROWS,INPUT_COLS,INPUT_CHANS, 1>,
    output_grad : &Buffer4D<T,1,INPUT_ROWS,INPUT_COLS,INPUT_CHANS>,
    activation : FusedActivation,
    strides: (usize, usize),
    padding : TensorViewPadding,

)->Buffer4D<T,1,WEIGHTS_ROWS,WEIGHTS_COLS,INPUT_CHANS>{
    let mut accum : Buffer4D<T,1,INPUT_ROWS,INPUT_COLS,INPUT_CHANS> =
        array::from_fn(|_|{SMatrix::from_fn(|_,_|[T::zero(); INPUT_CHANS])});
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in (0..INPUT_ROWS).step_by(strides.0){
        for output_col  in (0..INPUT_COLS).step_by(strides.1){
            for output_channel  in (0..INPUT_CHANS).step_by(strides.1){
                let val = outputs.buffer[0][(output_row,output_col)][output_channel];
                if !(match activation {
                    FusedActivation::Relu => {val > T::zero()},
                    FusedActivation::Relu6 => {val > T::zero() && val < quantized_6},
                    _ => true
                }){
                    continue;
                }
                let view : TensorView<T, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> = input.view((output_row,output_col),
                0,
                padding,
                strides);
                for filter_row in 0..WEIGHTS_ROWS{
                    for filter_cols in 0..WEIGHTS_COLS{
                        if view.mask[(filter_row,filter_cols)]{
                            accum[0][(filter_row,filter_cols)][output_channel] =
                                accum[0][(filter_row,filter_cols)][output_channel]
                                .saturating_add(view.buffer[(filter_row,filter_cols)][output_channel]
                                    .saturating_sub(input.zero_point[0]))
                                .saturating_mul(& output_grad[0][(output_row,output_col)][output_channel])
                                ;
                        }
                    }
                }
            }
        }
    }
    array::from_fn(|batch|{
        let filters_scale = 
            weights
            .scale
            .get(batch)
            .copied()
            .unwrap_or(weights.scale[0]);
        let scale = filters_scale * input.scale[0];
        SMatrix::from_fn(|i,j|{
            array::from_fn(|channel|{
                let tmp : f32 = T::to_superset(&accum[batch][(i,j)][channel]);
                T::from_superset(& (tmp * scale)).unwrap()
            })
        })
    }
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
>(
    input : &Tensor4D<T,1,INPUT_ROWS,INPUT_COLS,INPUT_CHANS, 1>,
    weights : &Tensor4D<T,1,WEIGHTS_ROWS,WEIGHTS_COLS,INPUT_CHANS, 1>,
    outputs : &Tensor4D<T,1,INPUT_ROWS,INPUT_COLS,INPUT_CHANS, 1>,
    output_grad : &Buffer4D<T,1,INPUT_ROWS,INPUT_COLS,INPUT_CHANS>,
    activation : FusedActivation,
    strides: (usize, usize),
    padding : TensorViewPadding,

)->Buffer4D<T,INPUT_CHANS,WEIGHTS_ROWS,WEIGHTS_COLS,INPUT_CHANS>{
    let mut accum : Buffer4D<T,1,INPUT_ROWS,INPUT_COLS,INPUT_CHANS> =
        array::from_fn(|_|{SMatrix::from_fn(|_,_|[T::zero(); INPUT_CHANS])});
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in (0..INPUT_ROWS).step_by(strides.0){
        for output_col  in (0..INPUT_COLS).step_by(strides.1){
            for output_channel  in (0..INPUT_CHANS).step_by(strides.1){
                let val = outputs.buffer[0][(output_row,output_col)][output_channel];
                if !(match activation {
                    FusedActivation::Relu => {val > T::zero()},
                    FusedActivation::Relu6 => {val > T::zero() && val < quantized_6},
                    _ => true
                }){
                    continue;
                }
                let coord = get_input_index(WEIGHTS_ROWS, WEIGHTS_COLS, (output_row,output_col), padding, strides);
                for filter_row in 0..WEIGHTS_ROWS{
                    if (coord.0 + filter_row as i32) < 0 {
                        continue;
                    }
                    for filter_col in 0..WEIGHTS_ROWS{
                        if (coord.1 + filter_col as i32) < 0{
                            continue;
                        }
                            let cur_coord = ((coord.0 + filter_row as i32) as usize, (coord.1 + filter_col as i32) as usize);
                            accum[0][cur_coord][output_channel] =
                                accum[0][cur_coord][output_channel]
                                .saturating_add(input.buffer[0][cur_coord][output_channel]
                                    .saturating_sub(weights.zero_point[0]))
                                .saturating_mul(& output_grad[0][(output_row,output_col)][output_channel])
                                ;
                            }
                    }
                }
            }
        }
    array::from_fn(|batch|{
        let filters_scale = 
            weights
            .scale
            .get(batch)
            .copied()
            .unwrap_or(weights.scale[0]);
        let scale = filters_scale * input.scale[0];
        SMatrix::from_fn(|i,j|{
            array::from_fn(|channel|{
                let tmp : f32 = T::to_superset(&accum[batch][(i,j)][channel]);
                T::from_superset(& (tmp * scale)).unwrap()
            })
        })
    }
    )
        
}
fn get_input_index(
        view_rows : usize,
        view_cols : usize,
        focus: (usize, usize),
        padding: TensorViewPadding,
        strides: (usize, usize),
    )->(i32, i32){
    match padding {
        TensorViewPadding::Same => {
            let shift = ((view_rows - 1) / 2, (view_cols - 1) / 2);
            (
                (strides.0 * focus.0) as i32 - (shift.0) as i32,
                (strides.1 * focus.1) as i32 - (shift.1) as i32
            )
        }
        TensorViewPadding::Valid => {
            ((strides.0 * focus.0) as i32, (strides.1 * focus.1) as i32)
        }
    }
                    
}
