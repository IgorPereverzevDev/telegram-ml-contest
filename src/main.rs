use linfa::*;
use linfa_bayes::*;
use linfa::traits::*;
use linfa_preprocessing::*;
use ndarray::*;
use polars::prelude::*;
use std::time::Instant;
use std::fs::File;
use ndarray_csv::Array2Reader;
use csv::ReaderBuilder;


fn main() {
    // Get current timestamp
    let now = Instant::now();

    // Data reading

    let file = File::open("data/train_data_5exp_labeled.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let df: Array2<u64> = reader.deserialize_array2((100, 2)).unwrap();

//    let df = CsvReader::from_path("data/train_data_5exp_labeled.csv")
//        .unwrap()
//        .finish()
//        .unwrap();
    // Selecting ["snippet", "language"] columns
//    let df = df
//        .select(["snippet", "language"])
//        .unwrap();
    // println!("Dataframe shape and sample:\n {:?}.", df.head(Some(1)));

    // Labels mapping data reading

    let file = File::open("data/labels.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let df: Array2<u64> = reader.deserialize_array2((100, 2)).unwrap();

//    let labels_vector : Vec<String> = CsvReader::from_path("data/labels.csv")
//        .unwrap()
//        .finish()
//        .unwrap()
//        .column("language")
//        .unwrap()
//        .utf8()
//        .unwrap()
//        .into_iter()
//        .map(|opt_s| opt_s.unwrap_or_default().to_string())
//        .collect();

    // Get vector of snippet column
    let snippets: Vec<String> = df
        .column("snippet")
        .unwrap()
        .utf8()
        .unwrap()
        .into_iter()
        .map(|opt_s| opt_s.unwrap_or_default().to_string())
        .collect();
    // Convert Vec<String> snippets to ArrayBase
    let snippets_array = Array::from(snippets);
    // IDK what is happening here:) And don't want..
    let vec_option_i64: Vec<Option<i64>> = df
        .column("language")
        .unwrap()
        .i64()
        .unwrap()
        .into_iter()
        .collect();
    let languages: Vec<usize> = vec_option_i64
        .into_iter()
        .map(|value| {value.unwrap() as usize})
        .collect();
    // Convert Vec languages to ArrayBase
    let languages_array = Array::from(languages);

    // TF-IDF vectorizer fit
    let vectorizer = tf_idf_vectorization::TfIdfVectorizer::default()
        // .document_frequency(2.0, 10000.0)
        .fit(&snippets_array)
        .unwrap();
    // println!("Vocabulary entries: {}.", vectorizer.nentries());
    // Snippets array to TF-IDF matrix transformation
    let tf_idf_matrix = vectorizer
        .transform(&snippets_array)
        .to_dense();
    // println!("TF-IDF matrix shape: {:?}.", tf_idf_matrix.shape());

    // Initialize dataset structure from tf-idf matrix and labels
    let dataset = DatasetBase::new(tf_idf_matrix, languages_array);

    // Naive bayes train
    let model = MultinomialNb::params().fit(&dataset).unwrap();

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
    let prediction_index = model
        .predict(&vectorized_test_snippet)[0];
    let prediction = labels_vector.get(prediction_index);
    let elapsed_inference = now_inference.elapsed();
    println!("\nSnippet: {}.", test_snippet);
    println!("Prediction: {:?}.", prediction);

    // Print execution time
    let elapsed = now.elapsed();
    println!("\nInference time: {:.2?}.", elapsed_inference);
    println!("Execution time: {:.2?}.", elapsed);
}
