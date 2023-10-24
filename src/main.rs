mod inference;
mod train;

pub fn main() {
    let test_snippet = "
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
        print('o commanded an shameless we disposing do.
               Indulgence ten remarkably nor are impression')";

    train::train_pipeline("data/train_data_5exp_labeled.csv", 0.08);
    inference::predict_pipeline(test_snippet);
}
