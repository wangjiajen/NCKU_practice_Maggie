# In[]
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
X_train = data['summary'] # Combine title and description (better accuracy than using them as separate features)
y_train = data['ClassIndex'].apply(lambda x: x-1).values # Class labels need to begin from 0
x_test = testdata['summary'] # Combine title and description (better accuracy than using them as separate features)
y_test = testdata['ClassIndex'].apply(lambda x: x-1).values # Class labels need to begin from 0

#Max Length of sentences in Train Dataset
maxlen = X_train.map(lambda x: len(x.split())).max()
data.head()

# In[]
data.shape, testdata.shape

# In[]
y_train = to_categorical(y_train,4)
y_test = to_categorical(y_test,4)

# In[]
vocab_size = 10000 # arbitrarily chosen
# Create and Fit tokenizer
tok = Tokenizer(num_words=vocab_size)
tok.fit_on_texts(X_train.values)

# Tokenize data
X_train = tok.texts_to_sequences(X_train)
x_test = tok.texts_to_sequences(x_test)

# Pad data
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# In[]
embed_dim = 100  # 嵌入向量總長度
num_heads = 2  # Number of attention heads
ff_dim = 100  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
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
# Shuffle the data
np.random.seed(42)
tf.random.set_seed(42)
# In[]
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history=model.fit(X_train, y_train, batch_size=512, epochs=10, validation_split=0.2)

# In[]
history.history
scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# In[]
prediction = model.predict(x_test)
labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']
for i in range(10,40,4):
    print(testdata['summary'].iloc[i][:50], "...")
    print("Actual category: ", labels[np.argmax(y_test[i])])
    print("predicted category: ",labels[np.argmax(prediction[i])])

# In[]
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from tensorflow.keras.utils import plot_model

y_test_arg=np.argmax(y_test,axis=1)
Y_pred = np.argmax(prediction,axis=1)
print(confusion_matrix(y_test_arg, Y_pred)) #y軸事實 x軸預測

from sklearn.metrics import classification_report
print(classification_report(y_test_arg, Y_pred))
# %%
