use libm::sinf;
use microflow::quantize::{self, quantize};
use microflow_train_macros::model;
use nalgebra::{matrix, SMatrix};
use ndarray::{Array3, Array4};
use ndarray_npy::read_npy;
use rand::Rng;
use std::{
    fs::{read_dir, File},
    io::Write,
};
#[model("models/train/sine.tflite", 1, "mse", false)]
struct Sine {}

fn main() {
    let mut model = Sine::new();
    let x = 0.5;
    let mut rng = rand::rng();
    let output_scale = 0.00785118155181408;
    let output_zero_point = 1i8;
    let epochs = 500;
    let samples = 1000;
    let batch = 50;
    let learning_rate = 0.01;
    let mut output_file = File::create("output_sine_untrained").unwrap();
    for sample in 0..samples {
        let x = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
        let y = sinf(x);
        let output = model.predict(matrix![x]);
        let output_string = format!("{} {} {}\n", x, y, output[0]);
        output_file.write_all(output_string.as_bytes()).unwrap();
    }
    println!("finished printing baseline");
    let mut counter = 0;
    println!("initial_weights: {}", model.weights0.buffer);
    'epochs: for e in 0..epochs {
        println!("epoch {}", e);
        for sample in 0..samples {
            let x = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
            // let y = 0.5 * sinf(x);
            let y = x / 2f32;
            // println!("x unquantized: {x}");
            // println!("y unquantized: {y}");
            let output = microflow::tensor::Tensor2D::quantize(
                matrix![y],
                [output_scale],
                [output_zero_point],
            );
            // println!("y quantized: {}", output.buffer[0]);

            let y_p = model.predict_train(matrix![x], &output, learning_rate)[0];
            // println!("predicted: {y_p}");
            // println!(
            //     "predicted quantized: {}",
            //     quantize(y_p, output_scale, output_zero_point)
            // );
            // println!("++++++++++++++++++++++++++++");
            if (sample + 1) % batch == 0 {
                // println!(
                //     "model batch gradient = {}",
                //     // learning_rate * model.weights0_gradient.cast::<f32>() / batch as f32
                //     model.weights0_gradient
                // );
                // panic!();
                // if counter == 3 {
                //     break 'epochs;
                // }
                counter += 1;
                model.update_layers(batch as usize, learning_rate);
            }
        }
    }
    println!("final_weights: {}", model.weights0.buffer);
    let mut output_file = File::create("output_sine_trained").unwrap();
    for sample in 0..samples {
        let x = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
        let y = 0.5 * sinf(x);
        let output = model.predict(matrix![x]);
        let output_string = format!("{} {} {}\n", x, y, output[0]);
        output_file.write_all(output_string.as_bytes()).unwrap();
    }
}
