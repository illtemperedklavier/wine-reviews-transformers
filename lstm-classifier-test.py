#!/usr/bin/env python
# coding: utf-8

#https://www.toptal.com/machine-learning/nlp-tutorial-text-classification


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Input, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
import jsonpickle as json
from util import input_maker

print("imported everything")

MAX_VOCAB_SIZE = 5000
VECTOR_DIM = 300
LATENT_DIM = 25

num_categories = 20

"""
make inputs
"""

inputs, outputs = input_maker.get_inputs_outputs(num_categories)
print("got inputs and outputs")


max_review_length = max(len(x) for x in inputs)
train_x, test_x, train_y, test_y = train_test_split(inputs, outputs, test_size=0.1)
dev_x, final_test_x, dev_y, final_test_y = train_test_split(test_x, test_y, test_size=0.3)

"""
make embedding matrix
"""

with open(r"D:\Data\wine-reviews\fasttext_vecs.json", "r") as f:
    fasttext_vectors = json.decode(f.read())
    
with open(r"D:\Data\wine-reviews\wine-words.pkl", 'rb') as f:
    words = pickle.load(f)

with open(r"D:\Data\wine-reviews\fdist1.pkl", "rb") as f:
    fdist1 = pickle.load(f)
    
words = [w for (w,_) in fdist1.most_common(MAX_VOCAB_SIZE)]

    
embeddings = np.zeros((MAX_VOCAB_SIZE+1, VECTOR_DIM))
idx2word = {}
word2idx = {}
i = 1


for w in range(MAX_VOCAB_SIZE):
    vec = fasttext_vectors.get(w)
    if vec is not None:
        vec = vec.astype(np.float_)
        embeddings[i] = vec
        idx2word[i] = w
        word2idx[w] = i
    else:
        vec = np.random.uniform(low = 0.0, high=2.0 , size = VECTOR_DIM)
        embeddings[i] = vec
        idx2word[i] = w
        word2idx[w] = i
    i+=1 


print("made embedding matrix")


embedding_layer = Embedding(
  MAX_VOCAB_SIZE+1,
  VECTOR_DIM,
  weights=[embeddings],
  # trainable=False
)


input_ = Input(shape=(max_review_length,))




"""
LSTM model 
"""


# create the model
#VECTOR_DIM

model = Sequential()
model.add(Embedding(MAX_VOCAB_SIZE + 1, VECTOR_DIM, weights=[embeddings], input_length=max_review_length))
model.add(LSTM(20))
model.add(Dense(num_categories, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=500, batch_size=64)
model.save(r"D:\Documents\Semantic-Health\lstm2.hdf5")


y_score = model.predict(test_x)
y_score = [[1 if i == max(sc) else 0 for i in sc] for sc in y_score]
n_right = 0

for i in range(len(y_score)):
    if all(y_score[i][j] == test_y[i][j] for j in range(len(y_score[i]))):
        n_right += 1

print("Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))
plt.plot(history.history['acc'])
list(history.history.keys())
