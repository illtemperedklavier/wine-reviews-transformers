import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from util import get_top_x
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TensorBoard

EMBEDDING_VECTOR_DIM = 64

data = pd.read_csv(r"D:\Data\wine-reviews\winemag-data-130k-v2.csv")

counter = Counter(data['variety'].tolist())
type(counter)

top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
data = data[data['variety'].map(lambda x: x in top_10_varieties)]
varietal_list = [top_10_varieties[i] for i in data['variety'].tolist()]
varietal_list = np.array(varietal_list)

description_list = data['description'].tolist()

varietal_list[:10]

"""
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(description_list)

x_train_counts.get_shape()
"""

mapped_list, word_list = get_top_x.filter_to_top_x(description_list, 2500, 10)

varietal_list_o = [top_10_varieties[i] for i in data['variety'].tolist()]

varietal_list = to_categorical(varietal_list_o)

max_review_length = max(len(x) for x in mapped_list)
train_x, test_x, train_y, test_y = train_test_split(mapped_list, varietal_list, test_size=0.3)

#callbacks
checkpoint = ModelCheckpoint('D:/Projects/Wine Reviews/checkpoints/weights_{epoch:02d}_{val_acc:.2f}.hdf5', verbose=1, save_best_only=True, mode='auto')


model = Sequential()

model.add(Embedding(2500, EMBEDDING_VECTOR_DIM, input_length=max_review_length))
model.add(Conv1D(50, 5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(max(varietal_list_o) + 1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=5, validation_split=0.1,batch_size=64)



#plt.plot(history.history['acc'])