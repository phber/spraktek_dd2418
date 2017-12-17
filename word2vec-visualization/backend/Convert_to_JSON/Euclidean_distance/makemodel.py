# coding: utf-8

import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import gensim


"""
Creating word2vec model from our data given in data.json
"""

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

df = pd.read_json('../../../data.json')

import string
stemmer = SnowballStemmer('swedish')
stop = stopwords.words('swedish') + list(string.punctuation.encode('utf-8')) + ['gt', 'lt', 'amp', 'quot', 'align', '**', '***', '--', '//', '://', '),', ').']

for i, s in enumerate(stop):
    try:
        stop[i] = s.replace(u'\xe5', 'aa').replace(u'\xe4', 'ae').replace(u'\xf6', 'oe')
    except AttributeError:
        pass

result = []
for doc in df['description']:
    sent = []
    for word in nltk.wordpunct_tokenize(doc.lower()):
        if word not in stop and not is_int(word):
            stemmed_word =  stemmer.stem(word)
            sent.append(word)
    result.append(sent)


model = gensim.models.Word2Vec(result, min_count=900, sg=1)
model.wv.save_word2vec_format('model.bin', binary=True)
