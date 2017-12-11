# coding: utf-8

import pandas as pd
import re, csv, nltk, time, json, itertools, os, nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
import gensim
from nltk.tokenize import RegexpTokenizer
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

def train(df):
    pass

def dendogram_plot(model):
    l = linkage(model.wv.syn0, method='complete', metric='seuclidean')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.ylabel('word')
    plt.xlabel('distance')

    dendrogram(
        l,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=16.,  # font size for the x axis labels
        orientation='left',
        leaf_label_func=lambda v: str(model.wv.index2word[v])
    )
    plt.show()

def tsne_plot(model):
    matplotlib.rcParams.update({'font.size': 11})
    tsne = TSNE(n_components=2)
    X = model[vocab]
    X_tsne = tsne.fit_transform(X)
    for i in range(len(vocab)):
        word = words[i]
        freq = model.wv.vocab[word].count
        x = X_tsne[:, 0][i]
        y = X_tsne[:, 1][i]
        plt.scatter(x, y, s=freq/10.0, cmap='viridis')
        plt.text(x, y, word)
    plt.show()

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

df = pd.read_json('output_new.json')

import string
stemmer = SnowballStemmer('swedish')
stop = stopwords.words('swedish') + list(string.punctuation.encode('utf-8')) + ['gt', 'lt', 'amp', 'quot', 'align', '**', '***', '--', '//', '://', '),', ').']

for i, s in enumerate(stop):
    stop[i] = s.replace(u'\xe5', 'aa').replace(u'\xe4', 'ae').replace(u'\xf6', 'oe')

result = []
for doc in df['description']:
    sent = []
    for word in nltk.wordpunct_tokenize(doc.lower()):
        if word not in stop and not is_int(word):
            stemmed_word =  stemmer.stem(word)
            sent.append(word)
    result.append(sent)

print df['description'].head()

ngram_vectorizer = CountVectorizer(ngram_range=(1,1), stop_words = stop, min_df=0.01)
counts = ngram_vectorizer.fit_transform(df['description'])
word_freq = counts.toarray().sum(axis=0)

model = gensim.models.Word2Vec(result, min_count=900)
vocab = model.wv.vocab
words = list(vocab)
print model
#tsne_plot(model)



"""
print 'Computing Distances'

from sklearn.metrics.pairwise import euclidean_distances
dist = euclidean_distances(counts.toarray().T)
vocab = ngram_vectorizer.get_feature_names()


from sklearn.manifold import MDS
print 'Running MDS'

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]

matplotlib.rcParams.update({'font.size': 9})
print 'Plotting'

for x, y, name, freq in zip(xs, ys, vocab,word_freq):
    size = freq/10.0
    plt.scatter(x, y, s = size)
    plt.text(x, y, name)

plt.show()
"""