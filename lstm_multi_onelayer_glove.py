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

import pandas as pd
import numpy as np
import re
from numpy import array
from numpy import asarray
from numpy import zeros

import matplotlib.pyplot as plt
from nltk.corpus import stopwords



def clean_text(text):
    text = text.lower().split()
    stop = set(stopwords.words("english"))
    text = [w for w in text if not w in stop and len(w) >= 3]
    text = " ".join(text)
    return text


df = pd.read_csv("multilabel.csv", error_bad_lines=False, header = 0, names=['Train_x', 'History', 'Diagnosis', 'Treatment', 'Other'])

df.dropna(how = "any", axis = 0, inplace=True)

diag_labels = df[['History', 'Diagnosis', 'Treatment', 'Other']]
diag_labels.head()

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size

X = []
sentences = list(df["Train_x"])
for sen in sentences:
    X.append(clean_text(sen))

y = diag_labels.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

maxlen = 10

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


embeddings_dictionary = dict()

glove_file = open('/Users/shravyag/Downloads/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((20000, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

input_1 = Input(shape=(maxlen,))
embedding_layer = Embedding(20000, 100, weights=[embedding_matrix], trainable=False)(input_1)
LSTM_Layer_1 = LSTM(100, dropout=0.5, recurrent_dropout=0.5)(embedding_layer)
dense_layer_1 = Dense(4, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=input_1, outputs=dense_layer_1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())

plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

history = model.fit(X_train, y_train, batch_size=128, epochs=9, verbose=1, validation_split=0.3)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()
