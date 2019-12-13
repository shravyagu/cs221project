from numpy import array

import pandas as pd
import numpy as np
import re
from numpy import array
from numpy import asarray
from numpy import zeros

from nltk.corpus import stopwords
#importing the glove library
from glove import Corpus, Glove
# creating a corpus object



def clean_text(text):
    text = text.lower().split()
    stop = set(stopwords.words("english"))
    text = [w for w in text if not w in stop and len(w) >= 3]
    return text


df = pd.read_csv("multilabel.csv", error_bad_lines=False, header = 0, names=['Train_x', 'History', 'Diagnosis', 'Treatment', 'Other'])

df.dropna(how = "any", axis = 0, inplace=True)

X = []
sentences = list(df["Train_x"])
for sen in sentences:
    X.append(clean_text(sen))
    #X.append(clean_text(sen))
#mydf = pd.DataFrame(X)

#mydf.to_csv('corpus.csv', index=False)

corpus = Corpus()
#training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(X, window=10)
#creating a Glove object which will use the matrix created in the above lines to create embeddings
#We can set the learning rate as it uses Gradient Descent and number of components
glove = Glove(no_components=100, learning_rate=0.05)

glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove.model')
