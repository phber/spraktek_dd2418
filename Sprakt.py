# coding: utf-8

import pandas as pd
import re, csv, nltk, time, json, itertools, os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
import gensim
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import Ridge, Lasso
import numpy as np
import string
import sklearn

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true = np.array(y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def regression(traindf, testdf, train_mtx, test_mtx):
    reg = Ridge()
    reg.fit(train_mtx,traindf['price'])
    predictions = reg.predict(test_mtx)
    test_result = testdf['price']
    print 'Mean Error:'
    print(sum(abs(predictions - test_result)) / len(predictions))
    average_price = sum(test_result)/len(test_result)
    print 'Baseline error:'
    print(sum(abs(average_price - test_result)) / len(predictions))
    print 'MAPE:'
    print mean_absolute_percentage_error(test_result, predictions)
    print 'R2:'
    print sklearn.metrics.r2_score(test_result, predictions)

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


def preprocessing(df):
    stop = stopwords.words('swedish') + list(string.punctuation.encode('utf-8')) + ['gt', 'lt', 'amp', 'quot', 'align', '**', '***', '--', '//', '://', '),', ').']
    for i, s in enumerate(stop):
        stop[i] = s.replace(u'\xe5', 'aa').replace(u'\xe4', 'ae').replace(u'\xf6', 'oe')
    result = []
    for i, row in df.iterrows():
        sent = []
        doc = row['description']
        for word in nltk.wordpunct_tokenize(doc.lower()):
            if word not in stop and not is_int(word):
                sent.append(word)
        sent = ' '.join(sent)
        result.append(sent)
    df['tokens'] = result


def ngram_calc(vocab, train, test, n = 1):
    ngram_vectorizer = CountVectorizer(ngram_range=(n,n), stop_words= 'english', min_df = 10)
    ngram_vectorizer.fit(vocab['tokens'])
    train_mtx = ngram_vectorizer.transform(train['description'])
    test_mtx = ngram_vectorizer.transform(test['description'])
    regression(train, test, train_mtx, test_mtx)


def tfid_calc(vocab, train, test, n = 1, regress = False):
    tf = TfidfVectorizer(ngram_range=(n,n), min_df = 10)
    counts = tf.fit(vocab['tokens'])
    train_mtx = tf.transform(train['tokens']).toarray()
    test_mtx = tf.transform(test['tokens']).toarray()
    print train_mtx.shape
    if regress:
        regression(train, test, train_mtx, test_mtx)
    return train_mtx, test_mtx

def run():
    vocab = pd.read_json('output_new.json')
    df = pd.read_json('sthlm_format.json')
    preprocessing(vocab)
    preprocessing(df)
    # Split into training and teast data
    msk = np.random.rand(len(df)) < 0.75
    train = df[msk]
    test = df[~msk]
    tfid_calc(vocab, train, test, 1, regress = True)

def word2vec_tfidf(model, df, tf_idfs):
    mtx = np.zeros((len(df.index),100))
    print df.shape, tf_idfs.shape
    docs = df['tokens'].tolist()
    for i in range(len(docs)):
        tokens = docs[i].split(' ')
        doc_score = np.zeros(100)
        for word in tokens:
            if word in model.wv.vocab:
                doc_score = doc_score + np.sum(tf_idfs[i,:])*model[word]
        doc_score = doc_score/len(tokens)
        mtx[i,:] = doc_score
    return mtx
    
def word2vec_calc(vocab, train, test):
    docs = [tokens.split(' ') for tokens in vocab['tokens']]
    model = gensim.models.Word2Vec(docs, min_count= 10)
    tfid_train, tfid_test = tfid_calc(vocab, train, test, 1)
    new_train = word2vec_tfidf(model, train, tfid_train)
    new_test = word2vec_tfidf(model, test, tfid_test)
    regression(train, test, new_train, new_test)

run()



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


