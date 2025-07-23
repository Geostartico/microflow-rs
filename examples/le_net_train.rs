use libm::sinf;
use microflow_train_macros::model;
use nalgebra::{matrix, Matrix, SMatrix};
use ndarray::{Array3, Array4};
use ndarray_npy::read_npy;
use rand::Rng;
use std::fs::{read_dir, File};
#[model("models/train/sine.tflite", 1, "mse", false)]
struct Sine {}
// #[model("models/train/lenet.tflite", 1, "mse", false)]
// struct LeNet {}
// #[model("models/train/speech.tflite", 1, "mse", false)]
// struct Speech {}

fn main() {
    let file = std::path::Path::new("examples/0.npy");
    let label_0: Vec<[SMatrix<[f32; 1], 28, 28>; 1]> =
        read_dir("datasets/fine_dataset_lenet/label_0")
            .unwrap()
            .filter(|el| el.is_ok())
            .map(|el| {
                let array: Array3<f32> = read_npy(el.unwrap().path()).unwrap();
                // println!(
                //     "{},{},{}",
                //     array.shape()[0],
                //     array.shape()[1],
                //     array.shape()[2],
                // );
                [SMatrix::from_fn(|i, j| [*array.get((i, j, 0)).unwrap()])]
            })
            .collect();

    let label_0: Vec<[SMatrix<[f32; 1], 125, 17>; 1]> =
        read_dir("datasets/dataset_fine_speech/label_0")
            .unwrap()
            .filter(|el| el.is_ok())
            .map(|el| {
                let array: Array3<f32> = read_npy(el.unwrap().path()).unwrap();
                println!(
                    "{},{},{}",
                    array.shape()[0],
                    array.shape()[1],
                    array.shape()[2],
                );
                [SMatrix::from_fn(|i, j| [*(array.get((i, j, 0)).unwrap())])]
            })
            .collect();
    let array: Array4<f32> = read_npy(file).unwrap();
    println!("{:?}", array);
    let mut model = Sine::new();
    let x = 0.5;
    let mut rng = rand::rng();
    let output_scale = 0.00826966855674982;
    let output_zero_point = 7i8;
    let y_predicted = model.predict(matrix![x])[0];
    let y_exact = sinf(x);
    let epochs = 5;
    let samples = 1000;
    let batch = 50;
    for _ in 0..epochs {
        for sample in 0..samples {
            let x = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
            let y = sinf(x);

            let output = microflow::tensor::Tensor2D::quantize(
                matrix![y],
                [output_scale],
                [output_zero_point],
            );

            model.predict_train(matrix![x], &output, 0.5);
            if sample % batch == 0 {
                model.update_layers(batch, 0.5);
            }
        }
    }
}
