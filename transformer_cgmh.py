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
class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        return ({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })


class TokenAndPositionEmbedding(layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size,
                                          output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        return ({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            # 'embedding_matrix': self.embedding_matrix
        })


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
train_df, test_df = train_test_split(
    data_filtered,
    test_size=test_split,
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

embed_dim = 100  # 嵌入向量總長度
num_heads = 2  # Number of attention heads
ff_dim = 100  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen, ))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(6, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
# %%
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=["accuracy"])
EPOCHS = 10
filepath = "transformer_cgmh.best.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                monitor='val_loss',
                                                mode='min',
                                                verbose=1,
                                                save_best_only=True)
callbacks_list = [checkpoint]
model.fit(X_train,
          y_train,
          batch_size=32,
          epochs=10,
          validation_split=0.05,
          callbacks=callbacks_list)
# %%
model.load_weights("transformer_cgmh.best.hdf5")
scores = model.evaluate(X_test, y_test)

# %%
