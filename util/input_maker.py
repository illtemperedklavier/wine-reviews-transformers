# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:32:42 2019

@author: alecr
"""

import jsonpickle as json
import pickle
import numpy as np
import pandas as pd
from util import get_top_x
#import get_top_x
from collections import Counter
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

MAX_VOCAB_SIZE = 5000 #chosen arbitrarily

with open(r"D:\Data\wine-reviews\fasttext_vecs.json", "r") as f:
    fasttext_vectors = json.decode(f.read())
    
    
    
with open(r"D:\Data\wine-reviews\wine-words.pkl", 'rb') as f:
    words = pickle.load(f)

"""   
vectors = []    
for w in words:
    vec = fasttext_vectors.get(w)
    if vec is not None:
        vectors.append(vec)
        
vectors = np.asarray(vectors)

"""



def get_inputs_outputs(num_most_common):
    df = pd.read_csv(r"D:\Data\wine-reviews\winemag-data-130k-v2.csv")

    counter = Counter(df['variety'].tolist())
    
    top_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(num_most_common))}
    df = df[df['variety'].map(lambda x: x in top_varieties)]
    
    #df['variety'].value_counts()
    
    description_list = df['description'].tolist()
    
    #create a list of the index of the wine type
    varietal_list = [top_varieties[i] for i in df['variety'].tolist()]
    varietal_list = np.array(varietal_list)
    
    
    varietal_list[:10]
    
    
    
    df = df[df['variety'].map(lambda x: x in top_varieties)]
    
    description_list = df['description'].tolist()
    
    
    mapped_list, word_list = get_top_x.filter_to_top_x(description_list, 2500, 10)
    
    
    varietal_list_o = [top_varieties[i] for i in df['variety'].tolist()]
    
    varietal_list = to_categorical(varietal_list_o)
    
    max_review_length = max(len(x) for x in mapped_list)
    mapped_list = pad_sequences(mapped_list, maxlen=max_review_length)
    
    
    return mapped_list, varietal_list


def get_raw_reviews():
    df = pd.read_csv(r"D:\Data\wine-reviews\winemag-data-130k-v2.csv")

    counter = Counter(df['variety'].tolist())
        
    top_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
    df = df[df['variety'].map(lambda x: x in top_varieties)]
        
    #df['variety'].value_counts()
        
    description_list = df['description'].tolist()
    return description_list

def get_BERT_inputs_outputs():
    df = pd.read_csv(r"D:\Data\wine-reviews\winemag-data-130k-v2.csv")

    counter = Counter(df['variety'].tolist())
    
    top_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
    df = df[df['variety'].map(lambda x: x in top_varieties)]
    
    #df['variety'].value_counts()
    
    description_list = df['description'].tolist()
    
    description_list = ["[CLS] " + sentence + " [SEP]" for sentence in description_list]
    
    #create a list of the index of the wine type
    varietal_list = [top_varieties[i] for i in df['variety'].tolist()]
    varietal_list = np.array(varietal_list)
    
    
    varietal_list[:10]
    
    
    
    #mapped_list, word_list = get_top_x.filter_to_top_x(description_list, 2500, 10)
    

    
    varietal_list = to_categorical(varietal_list)
    
    #max_review_length = max(len(x) for x in mapped_list)
    #mapped_list = pad_sequences(mapped_list, maxlen=max_review_length)
    
    
    return description_list, varietal_list