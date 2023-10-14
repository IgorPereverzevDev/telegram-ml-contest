use polars::prelude::*;
use linfa_preprocessing::*;
use linfa_bayes::*;
use ndarray::*;
use linfa::traits::*;
use linfa::metrics::*;
use linfa_datasets::*;


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

    // Get vector of snippet snippet column
    let snippets: Vec<String> = df.column("snippet")
        .unwrap()
        .utf8()
        .unwrap()
        .into_iter()
        .map(|opt_s| opt_s.unwrap_or_default().to_string())
        .collect();

    // Convert Vec<String> to ArrayBase
    let snippets_array = Array::from(snippets);

    // TF-IDF vectorizer fit
    let vectorizer = tf_idf_vectorization::TfIdfVectorizer::default()
        // .document_frequency(2.0, 10000.0)
        .fit(&snippets_array)
        .unwrap();

    println!("Vocabulary entries: {}.", vectorizer.nentries());

    // Snippets array to TF-IDF matrix transformation
    let tf_idf_matrix = vectorizer.transform(&snippets_array);

    println!("TF-IDF matrix shape: {:?}.", tf_idf_matrix.shape());

    // Naive Bayes test
    let (train, valid) = winequality()
        .map_targets(|x| if *x > 6 { "good" } else { "bad" })
        .split_with_ratio(0.9);
    let model = MultinomialNb::params().fit(&train).unwrap();
    let pred = model.predict(&valid);
    let cm = pred.confusion_matrix(&valid).unwrap();
    println!("{:?}", cm);
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());
}
