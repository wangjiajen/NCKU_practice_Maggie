# In[]
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers

from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

import os


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


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# In[]
data = pd.read_csv('/home/u108029050/m/data/train.csv')
testdata = pd.read_csv('/home/u108029050/m/data/test.csv')

#Set Column Names
data.columns = ['ClassIndex', 'Title', 'Description']
testdata.columns = ['ClassIndex', 'Title', 'Description']

data['summary'] = data['Title'] + ' ' + data['Description']
testdata['summary'] = testdata['Title'] + ' ' + testdata['Description']

data = data.drop(columns=['Title', 'Description'])
testdata = testdata.drop(columns=['Title', 'Description'])
#Combine Title and Description
X_data = data[
    'summary']  # Combine title and description (better accuracy than using them as separate features)
y_data = data['ClassIndex'].apply(
    lambda x: x - 1).values  # Class labels need to begin from 0
x_testdata = testdata[
    'summary']  # Combine title and description (better accuracy than using them as separate features)
y_testdata = testdata['ClassIndex'].apply(
    lambda x: x - 1).values  # Class labels need to begin from 0

#Max Length of sentences in Train Dataset
maxlen = X_data.map(lambda x: len(x.split())).max()
data.head()

# %%
# y_train = to_categorical(y_train,4)
# y_test = to_categorical(y_test,4)
max_words = 10000  # 僅考慮資料集中的前10000個單詞
maxlen = 100  # 100個文字後切斷評論
# Create and Fit tokenizer

tok = Tokenizer(num_words=max_words)  # 實例化一個只考慮最常用10000詞的分詞器
tok.fit_on_texts(X_data.values)  # 建構單詞索引
# vocab_size = len(tok.word_index) + 1

# 將文字轉成整數list的序列資料
X_data = tok.texts_to_sequences(X_data)
x_testdata = tok.texts_to_sequences(x_testdata)

# Pad data
X_data = keras.preprocessing.sequence.pad_sequences(X_data, maxlen=maxlen)
x_testdata = keras.preprocessing.sequence.pad_sequences(x_testdata,
                                                        maxlen=maxlen)

word_index = tok.word_index  #單詞和數字的字典
print('Found %s unique tokens' % len(word_index))
# print(len(X_train), "Training sequences")
# print(len(x_test), "Validation sequences")

# %%
print(X_data.shape)
print(x_testdata.shape)

# %%
training_samples = 96000  # We will be training on 10K samples
validation_samples = 24000  # We will be validating on 10000 samples
testing_samples = 7600
# Split data
X_train = X_data[:training_samples]
y_train = y_data[:training_samples]
X_val = X_data[training_samples:training_samples + validation_samples]
y_val = y_data[training_samples:training_samples + validation_samples]
X_test = x_testdata[:testing_samples]
y_test = y_testdata[:testing_samples]
# In[]
embed_dim = 100  # 嵌入向量總長度
num_heads = 2  # Number of attention heads
ff_dim = 100  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen, ))
embedding_layer = TokenAndPositionEmbedding(maxlen, max_words, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(4, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
# In[]
print(X_train.shape)
print(y_train.shape)
print(y_test.shape)
print(X_test.shape)
print(X_val.shape)
print(y_val.shape)
# In[]
# Shuffle the data
np.random.seed(42)
tf.random.set_seed(42)
# In[]
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
EPOCHS = 10
filepath = "transformer_agnews.best.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                monitor='val_loss',
                                                mode='min',
                                                verbose=1,
                                                save_best_only=True)
callbacks_list = [checkpoint]
model.fit(X_train,
          y_train,
          batch_size=512,
          epochs=EPOCHS,
          validation_data=(X_val, y_val),
          callbacks=callbacks_list)

# In[]
model.load_weights("transformer_agnews.best.hdf5")
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# In[]
prediction = model.predict(X_test)
labels = [
    'World News', 'Sports News', 'Business News', 'Science-Technology News'
]
for i in range(10, 40, 4):
    print(testdata['summary'].iloc[i][:50], "...")
    print("Actual category: ", labels[np.argmax(y_test[i])])
    print("predicted category: ", labels[np.argmax(prediction[i])])

# In[]

import sklearn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from tensorflow.keras.utils import plot_model
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

labels = [
    'World News', 'Sports News', 'Business News', 'Science-Technology News'
]
preds = [np.argmax(i) for i in prediction]
cm = confusion_matrix(y_test, preds)
plt.figure()
plt.rcParams.update({'font.size': 30})
plot_confusion_matrix(cm, figsize=(16, 12), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(4), labels, fontsize=15)
plt.yticks(range(4), labels, fontsize=15)
plt.show()
plt.savefig('test.png', bbox_inches="tight")
# print(sklearn.metrics.confusion_matrix(y_test,np.argmax(prediction, axis = 1), labels=None, sample_weight=None))
# print(sklearn.metrics.classification_report (y_test, np.argmax(prediction, axis = 1),label=[0,1]))

y_pred = np.argmax(prediction, axis=1)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1: {f1}")
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3]))

# %%
