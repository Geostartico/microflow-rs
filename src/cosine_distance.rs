use simba::scalar::{SubsetOf, SupersetOf};

use crate::buffer::Buffer4D;

pub fn cosine_similarity<
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
    T: SubsetOf<f32>,
    U: SubsetOf<f32>,
>(
    mat_a: Buffer4D<T, BATCHES, ROWS, COLS, CHANS>,
    mat_b: Buffer4D<T, BATCHES, ROWS, COLS, CHANS>,
) -> f32 {
    let mut scalar_prod = 0f32;
    let mut accum_a = 0f32;
    let mut accum_b = 0f32;
    for batch in 0..BATCHES {
        for row in 0..ROWS {
            for col in 0..COLS {
                for chan in 0..CHANS {
                    let el_a = f32::from_subset(&mat_a[batch][(row, col)][chan]);
                    let el_b = f32::from_subset(&mat_b[batch][(row, col)][chan]);
                    accum_a += el_a.powi(2);
                    accum_b += el_b.powi(2);
                    scalar_prod += el_a * el_b;
                }
            }
        }
    }
    scalar_prod / (accum_a.sqrt() * accum_b.sqrt())
}
