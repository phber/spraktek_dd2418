# coding: utf-8

import pandas as pd
import re, csv, nltk, time, json, itertools, os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
import gensim
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge, Lasso
import numpy as np
import string
import sklearn
from stop_words import get_stop_words
from TfidfEmbedding import TfidfEmbeddingVectorizer

stop_words = get_stop_words('swedish')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true = np.array(y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def regression(traindf, testdf, train_mtx, test_mtx, vocab = None):
    reg = Ridge()
    reg.fit(train_mtx,traindf['price']/100000)
    if vocab is not None: 
        coef = np.array(reg.coef_)
        top_ix = np.abs(coef).argsort()[-10:]
        print np.array(vocab)[top_ix], coef[top_ix]
    predictions = reg.predict(test_mtx)
    test_result = testdf['price']/100000
    guess_price = traindf['price']/100000
    mean_error =  (sum(abs(predictions - test_result)) / len(predictions))
    average_price = np.mean(guess_price)
    base_error = (sum(abs(average_price - test_result)) / len(predictions))
    r2_score = sklearn.metrics.r2_score(test_result, predictions)
    return mean_error, base_error, r2_score

def tsne_plot(model):
    matplotlib.rcParams.update({'font.size': 11})
    tsne = TSNE(n_components=2)
    X =  model[model.wv.vocab]
    X_tsne = tsne.fit_transform(X)
    for i in range(len(model.wv.vocab)):
        word, vec = model.wv.vocab.items()[i]
        freq = model.wv.vocab[word].count
        x = X_tsne[:, 0][i]
        y = X_tsne[:, 1][i]
        plt.scatter(x, y, s=freq/10.0)
        plt.text(x,y, word)
    plt.show()

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def preprocessing(df):
    stop = stopwords.words('swedish') + list(string.punctuation.encode('utf-8')) + ['ligger', 'finns', 'samt', 'a' 'gt', 'lt', 'amp', 'quot', 'align', '**', '***', '--', '//', '://', '),', ').']
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

def tfid_calc(vocab, train, test, n = 1, regress = False, split = 1):
    tf = TfidfVectorizer(ngram_range=(n,n), min_df = 1)
    msk = np.random.rand(len(vocab)) < split
    docs  = np.array(vocab['tokens'])[msk]
    counts = tf.fit(docs)
    train_mtx = tf.transform(train['tokens'])
    test_mtx = tf.transform(test['tokens'])
    if regress:
        return regression(train, test, train_mtx, test_mtx)
    return train_mtx, test_mtx, tf.get_feature_names()

def run(vocab_file, data_file):
    vocab = pd.read_json(vocab_file)
    df = pd.read_json(data_file)
    preprocessing(vocab)
    preprocessing(df)
    #print pd.Series(' '.join(df['tokens']).split()).value_counts()[:10]   
    means = []
    bases = []
    r2s = []
    for i in range (0,2):
        msk = np.random.rand(len(df)) < 0.75
        train = df[msk]
        test = df[~msk]
        print 'Run ' + str(i)
        mean_error, base_error, r2_score = tfid_calc(vocab, train, test, regress=True)
        #mean_error, base_error, r2_score = word2vec_calc(vocab, train, test)
        r2s.append(r2_score)
        means.append(mean_error)
        bases.append(base_error)
    print 'Mean error:'
    print str(np.mean(means)) + '+-' + str(np.std(means))
    print 'Base error:'
    print str(np.mean(bases)) + '+-' + str(np.std(bases))
    print 'R2 Score:'
    print str(np.mean(r2s)) + '+-' + str(np.std(r2s))
    plt.show()

def word2vec_calc(vocab, train, test, use_idf = False, split = 1):
    docs = np.array([tokens.split(' ') for tokens in vocab['tokens']])
    msk = np.random.rand(len(docs)) < split
    docs = docs[msk]
    model = gensim.models.Word2Vec(docs, min_count=5, size = 200, sg =1)
    word2idf = TfidfEmbeddingVectorizer(model)
    if use_idf:
        word2idf.fit(vocab['tokens'][msk])
    new_train = word2idf.transform([tokens.split(' ') for tokens in train['tokens']])
    new_test = word2idf.transform([tokens.split(' ') for tokens in test['tokens']])
    return regression(train, test, new_train, new_test)

#run('output_new.json', 'sthlm_format.json')