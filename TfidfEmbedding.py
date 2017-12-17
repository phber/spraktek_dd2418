from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict
import numpy as np


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = word2vec.vector_size

    def fit(self, X, y = None):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = np.exp(max(tfidf.idf_))
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, np.exp(tfidf.idf_[i])) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        if self.word2weight is None:
            return np.array([
                np.mean([self.word2vec[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
        else:
            return np.array([
                    np.mean([self.word2vec[w] * self.word2weight[w]
                            for w in words if w in self.word2vec] or
                            [np.zeros(self.dim)], axis=0)
                    for words in X
                ])

