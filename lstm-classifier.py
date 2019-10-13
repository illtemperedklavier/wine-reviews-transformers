#!/usr/bin/env python
# coding: utf-8


#https://www.toptal.com/machine-learning/nlp-tutorial-text-classification


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from util import get_top_x
import matplotlib.pyplot as plt


df = pd.read_csv(r"D:\Data\wine-reviews\winemag-data-130k-v2.csv")

counter = Counter(df['variety'].tolist())

type(counter)

top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
df = df[df['variety'].map(lambda x: x in top_10_varieties)]

df['variety'].value_counts()

description_list = df['description'].tolist()


varietal_list = [top_10_varieties[i] for i in df['variety'].tolist()]
varietal_list = np.array(varietal_list)
#note: this is making a list of the wine varieties by their index in the top 10

varietal_list[:10]



top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}


df = df[df['variety'].map(lambda x: x in top_10_varieties)]

description_list = df['description'].tolist()


mapped_list, word_list = get_top_x.filter_to_top_x(description_list, 2500, 10)


varietal_list_o = [top_10_varieties[i] for i in df['variety'].tolist()]


varietal_list = to_categorical(varietal_list_o)

max_review_length = max(len(x) for x in mapped_list)


mapped_list = sequence.pad_sequences(mapped_list, maxlen=max_review_length)
train_x, test_x, train_y, test_y = train_test_split(mapped_list, varietal_list, test_size=0.3)


embedding_vector_length = 64

model = Sequential()

model.add(Embedding(2500, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(50, 5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(max(varietal_list_o) + 1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=5, validation_split=0.1,batch_size=64)

y_score = model.predict(test_x)
y_score = [[1 if i == max(sc) else 0 for i in sc] for sc in y_score]
n_right = 0

for i in range(len(y_score)):
    if all(y_score[i][j] == test_y[i][j] for j in range(len(y_score[i]))):
        n_right += 1


print("Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))

plt.plot(history.history['acc'])


list(history.history.keys())


