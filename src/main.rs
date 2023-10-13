use linfa_trees::DecisionTree;
use linfa::prelude::*;
use linfa_datasets;

fn main() {
    println!("Hello, world!");
    // Load the dataset
    let dataset = linfa_datasets::iris();
    // Fit the tree
    let tree = DecisionTree::params().fit(&dataset).unwrap();
    // Get accuracy on training set
    let accuracy = tree.predict(&dataset).confusion_matrix(&dataset).unwrap().accuracy();

    println!("{}", accuracy);
}
