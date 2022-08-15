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

from transformer import TransformerBlock, TokenAndPositionEmbedding
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# In[]
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
# In[]
embed_dim = 100  # 嵌入向量總長度
num_heads = 2  # Number of attention heads
ff_dim = 100  # Hidden layer size in feed forward network inside transformer

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
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
EPOCHS = 10
filepath="transformer_agnews.best.hdf5"
checkpoint= tf.keras.callbacks.ModelCheckpoint(
     filepath,
     monitor='val_loss',
     mode='min',
     verbose=1,
     save_best_only=True)
callbacks_list = [checkpoint]
model.fit(X_train, y_train, batch_size=512, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=callbacks_list)

# In[]
model.load_weights("transformer_agnews.best.hdf5")
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# In[]
prediction = model.predict(X_test)
labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']
for i in range(10,40,4):
    print(testdata['summary'].iloc[i][:50], "...")
    print("Actual category: ", labels[np.argmax(y_test[i])])
    print("predicted category: ",labels[np.argmax(prediction[i])])

# In[]

import sklearn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from tensorflow.keras.utils import plot_model
print(sklearn.metrics.confusion_matrix(y_test,np.argmax(prediction, axis = 1), labels=None, sample_weight=None))
print(sklearn.metrics.classification_report (y_test, np.argmax(prediction, axis = 1)))# %%

# %%
