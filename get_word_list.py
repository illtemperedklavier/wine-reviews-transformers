# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:20:16 2019

@author: alecr
"""

import numpy as np
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import pickle
from nltk.corpus import stopwords


stopwords = stopwords.words('english')


df = pd.read_csv(r"D:\Data\wine-reviews\winemag-data-130k-v2.csv")

counter = Counter(df['variety'].tolist())

type(counter)

top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
df = df[df['variety'].map(lambda x: x in top_10_varieties)]

df['variety'].value_counts()

varietal_list = [top_10_varieties[i] for i in df['variety'].tolist()]
varietal_list = np.array(varietal_list)
#note: this is making a list of the wine varieties by their index in the top 10
varietal_list[:10]

top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
df = df[df['variety'].map(lambda x: x in top_10_varieties)]

descriptions = df['description']
punctuation = "-.,/?!—"
pun_trans = str.maketrans(punctuation," "*len(punctuation))

def add_to_word_list(s):
    s = s.lower()
    s = s.translate(pun_trans)
    tokens = word_tokenize(s)
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

word_list = []
for description in descriptions:
    word_list.extend(add_to_word_list(description))

print ("done")

fdist1 = FreqDist(word_list)


pickle.dump(fdist1, open())

def get_word_list(max_vocab_size):
    fdist1.most_common(max_vocab_size)
    new_words = [w for (w,_) in fdist1.most_common(max_vocab_size)]
    return new_words
    
    








#word_list = list(set(word_list))

#print ("done making word set")

#make frequency distribution of the words to choose the top n






"""
wine_words = open("D:\Data\wine-reviews\wine-words.pkl", "wb")
pickle.dump(word_list, wine_words)
wine_words.close()
"""






