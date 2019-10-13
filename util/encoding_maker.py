# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:32:42 2019

@author: alecr
"""

import jsonpickle as json
import pickle
import numpy as np


with open(r"D:\Data\wine-reviews\fasttext_vecs.json", "r") as f:
    fasttext_vectors = json.decode(f.read())
    
with open(r"D:\Data\wine-reviews\wine-words.pkl", 'rb') as f:
    words = pickle.load(f)
    
vectors = []    
for w in words:
    vec = fasttext_vectors.get(w)
    if vec is not None:
        vectors.append(vec)
        
vectors = np.asarray(vectors)

