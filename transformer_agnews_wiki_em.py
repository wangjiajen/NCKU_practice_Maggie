#In[]
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import tensorflow as tf
from keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
# %%
class TransformerBlock(layers.Layer): # Transformer的Encoder端，Transformer block塊
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att=layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn=keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
        self.layernorm1=layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2=layers.LayerNormalization(epsilon=1e-6)
        self.dropout1=layers.Dropout(rate)
        self.dropout2=layers.Dropout(rate)       
    def call(self, inputs, training):
        attn_output=self.att(inputs, inputs)
        attn_output=self.dropout1(attn_output, training=training)
        out1=self.layernorm1(inputs + attn_output)
        ffn_output=self.ffn(out1)
        ffn_output=self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
# %%
data = pd.read_csv('/home/u108029050/m/train.csv')
testdata = pd.read_csv('/home/u108029050/m/test.csv')

#Set Column Names 
data.columns = ['ClassIndex', 'Title', 'Description']
testdata.columns = ['ClassIndex', 'Title', 'Description']

data['summary'] = data['Title'] + ' ' + data['Description']
testdata['summary'] = testdata['Title'] + ' ' + testdata['Description']

data = data.drop(columns=['Title', 'Description'])
testdata = testdata.drop(columns=['Title', 'Description'])


#Combine Title and Description
X_data = data['summary'] # Combine title and description (better accuracy than using them as separate features)
y_data = data['ClassIndex'].apply(lambda x: x-1).values # Class labels need to begin from 0
x_testdata = testdata['summary'] # Combine title and description (better accuracy than using them as separate features)
y_testdata = testdata['ClassIndex'].apply(lambda x: x-1).values # Class labels need to begin from 0

#Max Length of sentences in Train Dataset
maxlen = X_data.map(lambda x: len(x.split())).max()
data.head()

# %%
# y_train = to_categorical(y_train,4)
# y_test = to_categorical(y_test,4)
max_words = 10000 # 僅考慮資料集中的前10000個單詞
maxlen = 100 # 100個文字後切斷評論
# Create and Fit tokenizer

tok = Tokenizer(num_words=max_words) # 實例化一個只考慮最常用10000詞的分詞器
tok.fit_on_texts(X_data.values) # 建構單詞索引
# vocab_size = len(tok.word_index) + 1

# 將文字轉成整數list的序列資料
X_data = tok.texts_to_sequences(X_data)
x_testdata = tok.texts_to_sequences(x_testdata)

# Pad data
X_data = keras.preprocessing.sequence.pad_sequences(X_data, maxlen=maxlen)
x_testdata = keras.preprocessing.sequence.pad_sequences(x_testdata, maxlen=maxlen)

word_index = tok.word_index #單詞和數字的字典
print('Found %s unique tokens' % len(word_index))
# print(len(X_train), "Training sequences")
# print(len(x_test), "Validation sequences")
# %%
print(X_data.shape)
print(x_testdata.shape)

# %%
training_samples = 96000  # We will be training on 10K samples
validation_samples = 24000  # We will be validating on 10000 samples
testing_samples=7600
# Split data
X_train = X_data[:training_samples]
y_train = y_data[:training_samples]
X_val = X_data[training_samples: training_samples + validation_samples]
y_val = y_data[training_samples: training_samples + validation_samples]
X_test =x_testdata[:testing_samples]
y_test =y_testdata[:testing_samples]
# %%
import os
embedding_index = {}
f = open('wiki.txt')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embedding_index[word] = coefs
    
f.close()

print('Found %s word vectors' % len(embedding_index))
print(embedding_index["google"])
# %%
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector         
# %%
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, max_words, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb=layers.Embedding(input_dim=max_words, output_dim=embed_dim,weights=[embedding_matrix],trainable=False)
        self.pos_emb=layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    def call(self, x):
        maxlen=tf.shape(x)[-1]
        positions=tf.range(start=0, limit=maxlen, delta=1)
        positions=self.pos_emb(positions)
        x=self.token_emb(x)
        return x + positions
# %%
embed_dim = 100  # 嵌入向量總長度
num_heads = 2  # Number of attention heads
ff_dim = 100  # Hidden layer size in feed forward network inside transformer
# %%
inputs = layers.Input(shape=(maxlen,))
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
# %%
print(X_train.shape)
print(y_train.shape)
print(y_test.shape)
print(X_test.shape)
print(X_val.shape)
print(y_val.shape)
# %%
# Shuffle the data
np.random.seed(42)
tf.random.set_seed(42)
# %%
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])
history=model.fit(X_train, y_train, batch_size=512, epochs=10, validation_data=(X_val, y_val))
# %%
history.history
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# %%
prediction = model.predict(X_test)
labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']
for i in range(10,40,4):
    print(testdata['summary'].iloc[i][:50], "...")
    print("Actual category: ", labels[np.argmax(y_test[i])])
    print("predicted category: ",labels[np.argmax(prediction[i])])
# %%
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from tensorflow.keras.utils import plot_model

y_test_arg=np.argmax(y_test,axis=1)
Y_pred = np.argmax(prediction,axis=1)
print(confusion_matrix(y_test_arg, Y_pred)) #y軸事實 x軸預測
# %%
from sklearn.metrics import classification_report
print(classification_report(y_test_arg, Y_pred)) 
# %%
