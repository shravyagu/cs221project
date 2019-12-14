# cs221project

# Extracting Medical Data From Unstructured Clinical Texts

This project seeks to use natural language process techniques in addition to neural networks to classify medical data into the categories of history, diagnosis, and treatment. This project was created for the purpose of Stanford University's CS 221 class, for the final project assignment.

## Required files:
The data used to train and test this code can be downloaded at https://www.kaggle.com/tboyle10/medicaltranscriptions/download.
The pre-trained glove embeddings that we utilized can be downloaded at https://www.kaggle.com/terenceliu4444/glove6b100dtxt.
The hand-labeled training set is included in this repo as multilabel.csv.

## Key packages:
We utilized Keras 2.3.1 and tensorflow 2.0.0 for all of our experiments.

## Code breakdown:
Each of the code files perform the following tasks:

- ff.py	: Uses a feed-forward neural network with Keras embeddings to do binary classification for each of the 3 main categories.
- k_means.py : Uses the k-means unsupervised learning algorithm to cluster the data into different algorithmically determined clusters.
- logistic.py : Uses logistic regression with Keras embeddings to do binary classification for each of the 3 main categories.
- lstm.py	Add : Uses RNNS with LSTMS with Keras embeddings to do binary classification for each of the 3 main categories.
- lstm_multi_multilayer_embedding.py	: Uses RNNS with LSTMS with Keras embeddings to do multilabel classification with multiple output layers for each of the 3 main categories.
- lstm_multi_multilayer_glove.py	: Uses RNNS with LSTMS with GloVe embeddings to do multilabel classification with multiple output layers for each of the 3 main categories.
- lstm_multi_onelayer_embedding.py	: Uses RNNS with LSTMS with Keras embeddings to do multilabel classification with a single output layer for the 3 main categories.
- lstm_multi_onelayer_glove.py : Uses RNNS with LSTMS with GloVe embeddings to do multilabel classification with a single output layer for the 3 main categories.

To run each of the files, simply ensure that the associated support files listed above are included in the same directory as the file you are attempting to run, and simply run the script. (No additional inputs are needed.)

## Citations and Acknowledgments
- Ahamed, S. (2019, March 27). Text Classification Using CNN, LSTM and Pre-trained Glove Word Embeddings: Part-3. Retrieved from https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa
- How to get accuracy, F1, precision and recall, for a keras model? (2019, February 6). Retrieved from https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
- Malik, U. (2019, August 27). Python for NLP: Multi-label Text Classification with Keras. Retrieved from https://stackabuse.com/python-for-nlp-multi-label-text-classification-with-keras/
- Visualization and Clustering in Python. (2019, August 19). Retrieved from https://stackoverflow.com/questions/57556818/visualization-and-clustering-in-python


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
