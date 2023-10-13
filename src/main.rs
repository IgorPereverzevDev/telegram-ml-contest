use polars::prelude::*;
use linfa_preprocessing::*;


fn main() {
    let _df = CsvReader::from_path("data/train_data.csv").unwrap().finish().unwrap();
    let _df = _df.select(["snippet", "language"]).unwrap();

    // println!("{}", _df.head(Some(3)));

    // This doesn't work, because to_ndarray doesn't convert strings.
    println!("{:?}", _df.to_ndarray::<Float64Type>(Default::default()).unwrap());

    // let x_train_counts = CountVectorizer::params()
    //     .n_gram_range(1,1)
    //     .fit(&_df.select(["snippet"])
    //         .unwrap()
    //         .to_ndarray::<Float64Type>(Default::default())
    //         .unwrap())
    //     .unwrap();
    //     // .transform(&_df.select(["snippet"]).unwrap());
    // println!("{}", x_train_counts.head(Some(3)));
}
