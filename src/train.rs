use linfa::*;
use linfa_bayes::*;
use linfa::traits::*;
use linfa_preprocessing::*;
use ndarray::*;
use ndarray_csv::*;
use csv::*;
use std::time::*;
use std::fs::*;
use bincode;

pub fn train_pipeline(train_file_path: &str, min_document_frequency: f32){
    // Get current timestamp
    let now = Instant::now();

    // Data reading
    let df : Array2<String> = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .from_reader(File::open(train_file_path)
            .unwrap())
        .deserialize_array2_dynamic()
        .unwrap();

    // Extracting snippets and languages from 2d array
    let snippets_array = df.slice(s![0, ..]);
    // Convert from String to i64
    let languages_array : Array1<usize> = df.slice(s![1, ..])
        .into_iter()
        .map(|x| x
            .to_string()
            .parse::<i64>()
            .unwrap() as usize)
        .collect();

    // TF-IDF vectorizer fit
    let vectorizer = tf_idf_vectorization::TfIdfVectorizer::default()
        .document_frequency(min_document_frequency, 1.0)
        .fit(&snippets_array)
        .unwrap();
    println!("TF-IDF fit: {:.2?}.", now.elapsed());
    // Snippets array to TF-IDF matrix transformation
    let tf_idf_matrix = vectorizer
        .transform(&snippets_array)
        .to_dense();
    println!("TF-IDF transform: {:.2?}.", now.elapsed());
    println!("TF-IDF matrix shape: {:?}.", tf_idf_matrix.shape());

    // Initialize dataset structure from tf-idf matrix and labels
    let dataset = DatasetBase::new(tf_idf_matrix, languages_array);

    // Naive bayes train
    let model = MultinomialNb::params()
        .fit(&dataset)
        .unwrap();
    println!("Naive bayes fit: {:.2?}.", now.elapsed());

    // Naive bayes test predict
    let test_snippet = "
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
        print('o commanded an shameless we disposing do.
               Indulgence ten remarkably nor are impression')";
    // Get current timestamp
    let now_inference = Instant::now();
    let test_snippet_array = Array::from(vec![test_snippet]);
    let vectorized_test_snippet = vectorizer
        .transform(&test_snippet_array)
        .to_dense();
    let prediction = model
        .predict(&vectorized_test_snippet)[0];
    let elapsed_inference = now_inference
        .elapsed();
    println!("\nSnippet: {}.", test_snippet);
    println!("Prediction: {}.", prediction);

    // Print execution time
    let elapsed = now
        .elapsed();
    println!("\nInference time: {:.2?}.", elapsed_inference);
    println!("Execution time: {:.2?}.", elapsed);

    //Saving trained tf-idf
    let _encoded: Vec<u8> = bincode::serialize(&vectorizer).unwrap();
    let mut file = File::create("artefacts/vectorizer.bin").unwrap();
    bincode::serialize_into(&mut file, &_encoded).unwrap();
    //Saving trained naive bayes
    let _encoded: Vec<u8> = bincode::serialize(&model).unwrap();
    let mut file = File::create("artefacts/nb_model.bin").unwrap();
    bincode::serialize_into(&mut file, &_encoded).unwrap();
}