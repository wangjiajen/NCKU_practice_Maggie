#In[]
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from tensorflow.keras import layers
from keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, classification_report
from tensorflow.keras.utils import plot_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from preprocessing import preprocess

maxlen, word_index, X_train, y_train, X_val, y_val, testdata, X_test, y_test = preprocess(
)
print("X_train set:", X_train.shape)
from transformer import TransformerBlock, TokenAndPositionEmbedding
#In[]

max_words = 10000
maxlen, word_index, X_train, y_train, X_val, y_val, testdata, X_test, y_test = preprocess(
    max_words=10000)

embedding_index = {}
f = open('wiki.txt')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

f.close()
#In[]
embed_dim = 100  # 嵌入向量總長度
num_heads = 2  # Number of attention heads
ff_dim = 100  # Hidden layer size in feed forward network inside transformer
# %%
inputs = layers.Input(shape=(maxlen, ))
embedding_layer = TokenAndPositionEmbedding(maxlen, max_words, embed_dim,
                                            embedding_matrix)
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
#In[]
# dataset = "agnews"
# assert dataset in ["agnews", "imdb"]
from tensorflow import keras
from keras_bert import get_custom_objects

model_path = "transformer_agnews_wiki_em.best.h5"
model.load_weights(model_path, by_name=False)
model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#In[]
prediction = model.predict(X_test)
labels = [
    'World News', 'Sports News', 'Business News', 'Science-Technology News'
]
for i in range(10, 40, 4):
    print(testdata['summary'].iloc[i][:50], "...")
    print("Actual category: ", labels[np.argmax(y_test[i])])
    print("predicted category: ", labels[np.argmax(prediction[i])])
#In[]
import sklearn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from keras.utils import plot_model

print(
    confusion_matrix(y_test,
                     np.argmax(prediction, axis=1),
                     labels=None,
                     sample_weight=None))
print(classification_report(y_test, np.argmax(prediction, axis=1)))
