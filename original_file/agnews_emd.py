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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
#   except RuntimeError as e:
#     print(e)

# %%
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
vocab_size = 10000 # 僅考慮資料集中的前10000個單詞
maxlen = 100 # 100個文字後切斷評論
# Create and Fit tokenizer

tok = Tokenizer(num_words=vocab_size) # 實例化一個只考慮最常用10000詞的分詞器
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
# %%
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

embedding_dim = 100

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))
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
EPOCHS = 10
filepath="agnews_em.best.hdf5"
checkpoint= tf.keras.callbacks.ModelCheckpoint(
     filepath,
     monitor='val_loss',
     mode='min',
     verbose=1,
     save_best_only=True)
callbacks_list = [checkpoint]
model.fit(X_train, y_train, batch_size=512, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=callbacks_list)
# %%
model.load_weights("agnews_em.best.hdf5")
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
import sklearn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from tensorflow.keras.utils import plot_model
print(sklearn.metrics.confusion_matrix(y_test,np.argmax(prediction, axis = 1), labels=None, sample_weight=None))
print(sklearn.metrics.classification_report (y_test, np.argmax(prediction, axis = 1)))
# y_test_arg=np.argmax(y_test,axis=1)
# Y_pred = model.predict(X_test)
# print(confusion_matrix(y_test_arg, Y_pred)) #y軸事實 x軸預測
# %%
# from sklearn.metrics import classification_report
# print(classification_report(y_test_arg, Y_pred)) 
# model.load_weights("weights.best.hdf5")
# model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])
# scores = model.evaluate(X_test, y_test)