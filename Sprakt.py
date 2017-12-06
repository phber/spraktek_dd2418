
# coding: utf-8

# In[233]:

import pandas as pd
import re, csv, nltk, time, langid, json, itertools, os, nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter


# In[143]:

nltk.download('stopwords')


# In[152]:

df = pd.read_json('C:/Users/Therese/Sprakt/output.json')


# In[153]:

df2 = df.copy()


# In[154]:

df2.head()


# In[155]:

df2['description'] = df2['description'].str.replace('\W+|\d+|[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]|[\u013a\u0123\u0142\u0155\u0117\u0101\u0137\u2026]' ,' ').str.lower()


# In[156]:

df2['description'].head()


# In[157]:

stemmer = SnowballStemmer('swedish')
stop = stopwords.words('swedish')


# In[158]:

wordlist = filter(None, " ".join(list(set(list(itertools.chain(*df2['description'].str.split(' ')))))).split(" "))


# In[211]:

wordlist


# In[159]:

df2['stemmed_text_data'] = [' '.join(filter(None,filter(lambda word: word not in stop, line))) for line in df2['description'].str.lower().str.split(' ')]


# In[171]:

minimum_count = 5
str_frequencies = pd.DataFrame(list(Counter(filter(None,list(itertools.chain(*df2['stemmed_text_data'].str.split(' '))))).items()),columns=['word','count'])
low_frequency_words = set(str_frequencies[str_frequencies['count'] < minimum_count]['word'])


# In[187]:

df2['stemmed_text_data'] = [' '.join(filter(None,filter(lambda word: word not in low_frequency_words, line))) for line in df2['stemmed_text_data'].str.split(' ')]
df2['stemmed_text_data'] = [" ".join(stemmer.stem(word) for word in next_text.split(' '))  for next_text in df2['stemmed_text_data']]


# In[193]:

texts_stemmed = filter(None, [next_text.strip(' ').split(' ') for next_text in df2['stemmed_text_data']])


# In[ ]:

w2vmodel_stemmed = gensim.models.Word2Vec(texts_stemmed, size=100, window=5, min_count=5, workers=4)
#w2vmodel_stemmed.save(savefolder+'w2v_stemmed_model')


# In[188]:

df2.head()


# In[229]:

v = TfidfVectorizer(stop_words = stop, min_df = 0.02, norm = 'l2')


# In[230]:

x = v.fit_transform(df2['description'])


# In[231]:

x.toarray().shape


# In[206]:

ngram_vectorizer = CountVectorizer(ngram_range=(2,2))


# In[207]:

counts = ngram_vectorizer.fit_transform(df2['description'])


# In[208]:

ngram_vectorizer.get_feature_names()


# In[209]:

counts


# In[ ]:



