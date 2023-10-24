# telegram-ml-contest
This repo contains solution for the task described in [telegram-ml-contest](https://contest.com/docs/ML-Competition-2023).

# Task definition

The task is to classify programming language by the code snippet. Time of execution for the solution should not exceed **10 ms** for a **4096** bytes snippet.

# Dataset

The dataset contains code snippet taken from the private repos on Github. Languages presented there include not only the most spread but also rare ones.


<img width="1107" alt="Screenshot 2023-10-13 at 20 52 13" src="https://github.com/IgorPereverzevDev/telegram-ml-contest/assets/72782370/1842ad35-6ba3-4b70-9bba-f9df561a4d5f">

More detailed information on the content may be found on Kaggle.

# Solution

The taks is solved with the combination of Naive Bayes with TF-IDF vectorizer. For sake of execution time the final version were written using Rust instead of Python.

## Multinomial Naive Bayes

The family of Naive Bayes classifiers assume independence between variables. They do not model moments between variables and lack therefore in modelling capability. The advantage is a linear fitting time with maximum-likelihood training in a closed form. Linfa bayes were used in this solution, more info you will find [here](https://docs.rs/linfa-bayes/latest/linfa_bayes/struct.MultinomialNb.html#model-usage-example)

## TF-IDF Vectorizer

Simlar to CountVectorizer but instead of just counting the term frequency of each vocabulary entry in each given document, it computes the term frequecy times the inverse document frequency, thus giving more importance to entries that appear many times but only on some documents. The weight function can be adjusted by setting the appropriate method. More information [here](https://docs.rs/linfa-preprocessing/latest/linfa_preprocessing/tf_idf_vectorization/index.html).

 ## Execution

 The dataset snippets get transformed with TF-IDF vectorizer then the transformed dataset delivered into Multinomial Naive Bayes which creates final prediction. 

 ## Library

 The final model is packed as shareble object wich can be plugged into various services.
