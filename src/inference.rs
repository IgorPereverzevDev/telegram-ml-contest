use std;
use ndarray::*;
use linfa_bayes::*;
use linfa::traits::*;
use linfa_preprocessing::*;
use bincode;


pub fn predict_pipeline(snippet: &str) -> i64 {
    // Initialize tf-idf trained model
    let mut f = std::fs::File::open("artefacts/vectorizer.bin")
        .unwrap();
    let _decoded: Vec<u8> = bincode::deserialize_from(&mut f)
        .unwrap();
    let vectorizer: tf_idf_vectorization::FittedTfIdfVectorizer = bincode::deserialize(&_decoded)
        .unwrap();
    // Initialize naive bayes trained model
    let mut f = std::fs::File::open("artefacts/nb_model.bin")
        .unwrap();
    let _decoded: Vec<u8> = bincode::deserialize_from(&mut f)
        .unwrap();
    let model:  MultinomialNb<f64, usize> = bincode::deserialize(&_decoded)
        .unwrap();

    // Snippet string to array
    let test_snippet_array = Array::from(vec![snippet]);
    // Snippet vectorization
    let vectorized_test_snippet = vectorizer
        .transform(&test_snippet_array)
        .to_dense();
    // Prediction
    let prediction = model
        .predict(&vectorized_test_snippet)[0];
    prediction as i64
}
