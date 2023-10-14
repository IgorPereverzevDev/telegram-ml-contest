use polars::prelude::*;
use linfa_preprocessing::*;
use ndarray::Array;


fn main() {
    // Data reading
    let df = CsvReader::from_path("data/train_data_5exp.csv")
        .unwrap()
        .finish()
        .unwrap();
    // Selecting ["snippet", "language"] columns
    let df = df.select(["snippet", "language"])
        .unwrap();

    println!("Dataframe shape: {:?}.", df.shape());

    // Get vector of snippet df column
    let snippets: Vec<String> = df.column("snippet")
        .unwrap()
        .utf8()
        .unwrap()
        .into_iter()
        .map(|opt_s| opt_s.unwrap_or_default().to_string())
        .collect();


    // Convert Vec<String> to ArrayBase
    let snippets_array = Array::from(snippets);

    // Now use the array for CountVectorizer
    let count_vectorizer = CountVectorizer::params()
        .n_gram_range(1, 1)
        .fit(&snippets_array)
        .unwrap();

    // Transform snippets_array by count_vectorizer
    let training_matrix = count_vectorizer
        .transform(&snippets_array);

    println!("{:?}", training_matrix.shape())
}
