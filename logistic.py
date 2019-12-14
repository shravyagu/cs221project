from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.utils import plot_model
import keras.backend as K

import pandas as pd
import numpy as np
import re
from numpy import array
from numpy import asarray
from numpy import zeros

import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# Create custom metric for f1_score.
def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    #Fraction of relevant items that are selected.
    recall =  K.switch(K.not_equal(c3, 0), K.cast_to_floatx(c1/c3), K.cast_to_floatx(0.))
    # Fraction of selected items that are relevant.
    precision = K.switch(K.not_equal(c2, 0), K.cast_to_floatx(c1/c2), K.cast_to_floatx(0.))

    # F1_score
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.cast_to_floatx(f1_score)

# Create custom metric for precision.
def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))

    # Fraction of selected items that are relevant.
    precision = K.switch(K.not_equal(c2, 0.), K.cast_to_floatx(c1/c2), K.cast_to_floatx(0.))
    return precision

# Create custom metric for recall.
def recall(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # Fraction of relevant items that are selected.
    recall = K.switch(K.not_equal(c3, 0.), K.cast_to_floatx(c1/c3), K.cast_to_floatx(0.))
    return recall

# Clean the data by removing stopwords, unifying sentence case, removing words that have a length less than 3.
def clean_text(text):
    text = text.lower().split()
    stop = set(stopwords.words("english"))
    text = [w for w in text if not w in stop and len(w) >= 3]
    text = " ".join(text)
    return text

df = pd.read_csv("multilabel.csv", error_bad_lines=False, header = 0, names=['Train_x', 'History', 'Diagnosis', 'Treatment', 'Other'])

# Remove any empty rows and perform all cleaning.
df.dropna(how = "any", axis = 0, inplace=True)
for row in range(len(df)):
    try:
        mystr = clean_text(str(df['Train_x'].iloc[row]))
        df['Train_x'].iloc[row] = mystr
    except:
        pass

# Create multiple independent dataframes for each of the 4 categories.
hist_df = df[["Train_x", "History"]]
diag_df = df[["Train_x", "Diagnosis"]]
treat_df = df[["Train_x", "Treatment"]]

# Start with the History dataset.
labels = df["History"]
# Convert multilabel dataset to binary classification problem.
for row in range(len(df)):
    if(df['History'].iloc[row] == 1):
        hist_df['History'].iloc[row] = 1
    else:
        hist_df['History'].iloc[row] = 0

# Split the dataset into training and testing data.
X_list = hist_df["Train_x"]
X_arr = np.asarray(X_list)
X = X_arr.transpose()
y = labels.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Tokenize the dataset.
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
maxlen = 50
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# Perform Logistic Regression through one layer with just a sigmoid activation.
model = Sequential()
model.add(Embedding(20000, 128, input_length=50))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc', f1_score, precision, recall])

plot_model(model, to_file='hist_logist.png', show_shapes=True, show_layer_names=True)

history = model.fit(X_train, y_train, batch_size=100, epochs=1, verbose=1, validation_split=0.2)
print(model.metrics_names)
print(model.evaluate(X_test, y_test, verbose=1))

#####################################################################################################################################################################################
# Move to diagnosis dataframe.


labels = df["Diagnosis"]
# Convert multilabel dataset to binary classification problem.
for row in range(len(df)):
    if(df['Diagnosis'].iloc[row] == 1):
        diag_df['Diagnosis'].iloc[row] = 1
    else:
        diag_df['Diagnosis'].iloc[row] = 0

# Split the dataset into training and testing data.
X = diag_df["Train_x"]
y = labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Tokenize the dataset.
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
maxlen = 50
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# Perform Logistic Regression through one layer with just a sigmoid activation.
model = Sequential()
model.add(Embedding(20000, 128, input_length=50))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc', f1_score, precision, recall])


plot_model(model, to_file='diag_logist.png', show_shapes=True, show_layer_names=True)

history = model.fit(X_train, y_train, batch_size=100, epochs=1, verbose=1, validation_split=0.2)

print(model.metrics_names)
print(model.evaluate(X_test, y_test, verbose=1))


#####################################################################################################################################################################################
# Move to treatment dataframe.

labels = df["Treatment"]
# Convert multilabel dataset to binary classification problem.
for row in range(len(df)):
    if(df['Treatment'].iloc[row] == 1):
        treat_df['Treatment'].iloc[row] = 1
    else:
        treat_df['Treatment'].iloc[row] = 0

# Split the dataset into training and testing data.
X = treat_df["Train_x"]
y = labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Tokenize the dataset.
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
maxlen = 50
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# Perform Logistic Regression through one layer with just a sigmoid activation.
model = Sequential()
model.add(Embedding(20000, 128, input_length=50))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc', f1_score, precision, recall])

plot_model(model, to_file='treat_logist.png', show_shapes=True, show_layer_names=True)

history = model.fit(X_train, y_train, batch_size=100, epochs=1, verbose=1, validation_split=0.2)

print(model.metrics_names)
print(model.evaluate(X_test, y_test, verbose=1))
