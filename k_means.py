import collections
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import pandas as pd
import random
import matplotlib.pyplot as plt

def word_tokenizer(text):
    #tokenizes and stems the text
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens

def cluster_sentences(sentences, nb_of_clusters=7):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
        stop_words=stopwords.words('english'), max_df=0.9,
        min_df=0.1, lowercase=True)
    #builds a tf-idf matrix for the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=nb_of_clusters)
    kmeans.fit(tfidf_matrix)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    # Plot the clisters and number of assigned values.
    plt.hist(kmeans.labels_, bins=nb_of_clusters)
    plt.show()
    return (dict(clusters))

if __name__ == "__main__":
    df = pd.read_csv("records.csv")
    sentences = []
    df_short = df[0:3000]
    # Prepare data for appropriate clustering and cluster.
    for i,row in df_short.iterrows():
        mytranscript = row['transcription']
        try:
            sentences += mytranscript.split('.')
        except:
            pass
    nclusters= 20
    clusters = cluster_sentences(sentences, nclusters)


    for i,cluster in enumerate(list(clusters.keys())):
        mysentindices = clusters[cluster]
        sents = []
        for ind in mysentindices:
            sents.append(sentences[ind])

        # Print out some samples from each of the samples.
        mysam = random.sample(sents, 10)
        for sam in mysam:
            print("My Cluster : " + str(cluster) + ". Sentences " + str(sam) + "\n")
