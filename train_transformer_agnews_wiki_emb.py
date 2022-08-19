#In[]
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer

from transformer import TransformerBlock, TokenAndPositionEmbedding
from preprocessing import preprocess
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

f.close()

print('Found %s word vectors' % len(embedding_index))
print(embedding_index["google"])

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

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

# %%
# Shuffle the data
np.random.seed(42)
tf.random.set_seed(42)

model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
filepath = "transformer_agnews_wiki_em.best.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                monitor='val_loss',
                                                mode='min',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(X_train,
          y_train,
          batch_size=512,
          epochs=1,
          validation_data=(X_val, y_val),
          callbacks=callbacks_list)
model.save_weights("transformer_agnews_wiki_em.best.h5")