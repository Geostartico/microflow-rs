use microflow::buffer::Buffer2D;
use microflow_train_macros::model;

#[path = "../samples/features/person_detect.rs"]
mod features;

#[model("models/person_detect.tflite", 10)]
struct PersonDetect{}

fn print_prediction(prediction: Buffer2D<f32, 1, 2>) {
    println!(
        "Prediction: {:.1}% no person, {:.1}% person",
        prediction[0] * 100.,
        prediction[1] * 100.,
    );
    println!(
        "Outcome: {}",
        match prediction.iamax_full().1 {
            0 => "NO PERSON",
            1 => "PERSON",
            _ => unreachable!(),
        }
    );
}

fn main() {
    let model = PersonDetect::new();
    let person_predicted = model.predict_quantized(features::PERSON);
    let no_person_predicted = model.predict_quantized(features::NO_PERSON);
    println!();
    println!("Input sample: 'person.bmp'");
    print_prediction(person_predicted);
    println!();
    println!("Input sample: 'no_person.bmp'");
    print_prediction(no_person_predicted);
}
