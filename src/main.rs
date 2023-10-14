use linfa::*;
use linfa_bayes::*;
use linfa::traits::*;
use linfa::metrics::*;
use linfa_preprocessing::*;
use ndarray::*;
use polars::prelude::*;


fn main() {
    // Data reading
    let df = CsvReader::from_path("data/train_data_5exp_labeled.csv")
        .unwrap()
        .finish()
        .unwrap();
    // Selecting ["snippet", "language"] columns
    let df = df.select(["snippet", "language"])
        .unwrap();
    println!("Dataframe shape and sample:\n {:?}.", df.head(Some(1)));

    // Get vector of snippet column
    let snippets: Vec<String> = df.column("snippet")
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
    println!("Vocabulary entries: {}.", vectorizer.nentries());
    // Snippets array to TF-IDF matrix transformation
    let tf_idf_matrix = vectorizer.transform(&snippets_array).to_dense();
    println!("TF-IDF matrix shape: {:?}.", tf_idf_matrix.shape());

    // Initialize dataset structure from tf-idf matrix and labels
    let dataset = DatasetBase::new(tf_idf_matrix, languages_array);

    // Naive bayes train
    let model = MultinomialNb::params().fit(&dataset).unwrap();
    let pred = model.predict(&dataset);
    let cm = pred.confusion_matrix(&dataset).unwrap();
    println!("Accuracy {}%!", cm.accuracy() * 100.0);
}
