# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:13:09 2019

@author: alecr

For this, I'm only loading the vectors for words which appear in the dataset, otherwise, there's just too many for my RAM


"""

import io
import pickle
import numpy as np

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data
        

with open("D:\Data\wine-reviews\wine-words.pkl", "rb") as f:
    words = pickle.load(f)
    


path = r"D:\Data\FastText\wiki-news-300d-1M.vec\wiki-news-300d-1M.vec"

fasttext_wine = {}

total = 0
fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
line = fin.readline() #this line is a header, I can ignore it
total+=1
while line:
    line = fin.readline()
    total+=1
    line = line.split(' ')
    word = line[0]
    vector = line[1:]
    if word in words:
        fasttext_wine[word] = np.asarray(vector)  
    
    
    if total % 100000 == 0:
        print("at line %d" % total )
        

print('done')

