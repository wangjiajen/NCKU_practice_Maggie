#In[]
from tkinter import Y
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from sklearn.model_selection import train_test_split
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#In[]
np.random.seed(42)
tf.random.set_seed(42)
data = pd.read_csv('/home/u108029050/m/data/cgmh_six_classes.csv')
data.head()
#In[]
print(f"There are {len(data)} rows in the dataset.")
#In[]
print(sum(data["y_true"].value_counts() == 1))
print(data["y_true"].nunique())
#In[]
# Filtering the rare terms.
data_filtered = data.groupby("y_true").filter(lambda x: len(x) > 1)
data_filtered.shape


#In[]
# y_true = []
# for string_y in data_filtered["y_true"]:
#     string_labels = string_y[1:-1].split(",")
#     int_labels = [int(s) for s in string_labels]
#     y_true.append(int_labels)
# data_filtered["y_true"] = y_true
def string_toint(x):
    string_labels = x[1:-1].split(",")
    int_labels = [int(s) for s in string_labels]
    return int_labels


data_filtered["y_true"] = data_filtered["y_true"].apply(string_toint)
# %%# %%
test_split = 0.05
# Initial train and test split.
train_df, test_df = train_test_split(data_filtered
                                     # ,test_size=test_split,
                                     )
# 將stratify=X就是按照X中的比例分配，將stratify=y就是按照y中的比例分配
print(f"Number of rows in training set: {len(train_df)}")
print(f"Number of rows in test set: {len(test_df)}")

X_train = train_df['X']
y_train = train_df['y_true'].tolist()
X_test = test_df['X']
y_test = test_df['y_true'].tolist()

# %%
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

vocab_size = 5000
maxlen = 150
tok = Tokenizer(num_words=vocab_size)  # 實例化一個只考慮最常用10000詞的分詞器
tok.fit_on_texts(X_train.values)  # 建構單詞索引
tok.fit_on_texts(X_test.values)  # 建構單詞索引
X_train = tok.texts_to_sequences(X_train)
X_test = tok.texts_to_sequences(X_test)
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)
word_index = tok.word_index
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
print('Found %s unique tokens' % len(word_index))
# %%
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

embedding_dim = 100
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()
# %%
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=["accuracy"])
EPOCHS = 10
filepath = "cgmh_emb.best.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                monitor='val_loss',
                                                mode='min',
                                                verbose=1,
                                                save_best_only=True)
callbacks_list = [checkpoint]
model.fit(X_train,
          y_train,
          batch_size=8,
          epochs=10,
          validation_split=0.05,
          callbacks=callbacks_list)
# %%
model.load_weights("cgmh_emb.best.hdf5")
scores = model.evaluate(X_test, y_test)

# %%
