#In[]
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras 
from keras import layers
# %%
import os
class TransformerBlock(layers.Layer): # Transformer的Encoder端，Transformer block塊
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
      super(TransformerBlock, self).__init__()
      self.embed_dim = embed_dim
      self.num_heads = num_heads
      self.ff_dim = ff_dim
      self.rate = rate
      self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
      self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
      self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
      self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
      self.dropout1 = layers.Dropout(rate)
      self.dropout2 = layers.Dropout(rate)       
    def call(self, inputs, training):
        attn_output=self.att(inputs, inputs)
        attn_output=self.dropout1(attn_output, training=training)
        out1=self.layernorm1(inputs + attn_output)
        ffn_output=self.ffn(out1)
        ffn_output=self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
     config = super().get_config().copy()
     config.update({'embed_dim':self.embed_dim,
          'num_heads':self.num_heads,
          'ff_dim':self.ff_dim,
          'rate':self.rate})
     return config
# %%
imdb_dir = '/home/u108029050/m/data/aclImdb' # Data directory
train_dir = os.path.join(imdb_dir, 'train') # Get the path of the train set
test_dir = os.path.join(imdb_dir, 'test') # Get the path of the train set
# Setup empty lists to fill
labels_train = []
texts_train = []
labels_test = []
texts_test = []
# First go through the negatives, then through the positives
for label_type in ['neg', 'pos']:
    # Get the sub path
    dir_name = os.path.join(train_dir, label_type)
    print('loading train ',label_type)
    # Loop over all files in path
    for fname in tqdm(os.listdir(dir_name)):
        
        # Only consider text files
        if fname[-4:] == '.txt':
            # Read the text file and put it in the list
            f = open(os.path.join(dir_name, fname))
            texts_train.append(f.read())
            f.close()
            # Attach the corresponding label
            if label_type == 'neg':
                labels_train.append(0)
            else:
                labels_train.append(1)
for label_type in ['neg', 'pos']:
    # Get the sub path
    dir_name = os.path.join(test_dir, label_type)
    print('loading test',label_type)
    # Loop over all files in path
    for fname in tqdm(os.listdir(dir_name)):
        
        # Only consider text files
        if fname[-4:] == '.txt':
            # Read the text file and put it in the list
            f = open(os.path.join(dir_name, fname))
            texts_test.append(f.read())
            f.close()
            # Attach the corresponding label
            if label_type == 'neg':
                labels_test.append(0)
            else:
                labels_test.append(1)
# %%
from keras.preprocessing.text import Tokenizer
import numpy as np

max_words = 10000 # We will only consider the 10K most used words in this dataset

tokenizer = Tokenizer(num_words=max_words) # Setup
tokenizer.fit_on_texts(texts_train)
tokenizer.fit_on_texts(texts_test) # Generate tokens by counting frequency
sequences_test = tokenizer.texts_to_sequences(texts_train) 
sequences_train = tokenizer.texts_to_sequences(texts_test)# Turn text into sequence of numbers
word_index = tokenizer.word_index
# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences
maxlen = 100 # Make all sequences 100 words long
data_test = pad_sequences(sequences_test, maxlen=maxlen)
data_train = pad_sequences(sequences_train, maxlen=maxlen)
labels_train = np.asarray(labels_train)
labels_test = np.asarray(labels_test)
# %%
# Shuffle data
#In[]
training_samples = 20000  # We will be training on 10K samples
validation_samples = 5000  # We will be validating on 10000 samples
testing_samples=25000
# Split data
X_train = data_train[:training_samples]
y_train =labels_train[:training_samples]
X_val = data_train[training_samples: training_samples + validation_samples]
y_val = labels_train[training_samples: training_samples + validation_samples]
X_test =data_test[:testing_samples]
y_test =labels_test[:testing_samples]

print(X_train.shape)
print(y_train.shape)
print(y_test.shape)
print(X_test.shape)
print(X_val.shape)
print(y_val.shape)
#In[]
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
#In[]
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
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb=layers.Embedding(input_dim=max_words, output_dim=embed_dim,weights=[embedding_matrix],trainable=False)
        self.pos_emb=layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    def call(self, x):
        maxlen=tf.shape(x)[-1]
        positions=tf.range(start=0, limit=maxlen, delta=1)
        positions=self.pos_emb(positions)
        x=self.token_emb(x)
        return x + positions
    def get_config(self):
      config = super().get_config().copy()
      config.update({
        'maxlen': self.maxlen,
        'vocab_size': self.vocab_size,
        'embed_dim': self.embed_dim
       })
      return config
# %%
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
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
outputs = layers.Dense(2, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
# %%
np.random.seed(42)
tf.random.set_seed(42)
#In[]

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
EPOCHS = 10
filepath="transformer_IMDB_wiki_em.best.hdf5"
checkpoint= tf.keras.callbacks.ModelCheckpoint(
     filepath,
     monitor='val_loss',
     mode='min',
     verbose=1,
     save_best_only=True)
callbacks_list = [checkpoint]
model.fit(X_train, y_train, batch_size=16, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=callbacks_list)

# In[]

model.load_weights("transformer_IMDB_wiki_em.best.hdf5")
scores = model.evaluate(X_test, y_test)
# # In[]
prediction = model.predict(X_test)
import sklearn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from tensorflow.keras.utils import plot_model
print(sklearn.metrics.confusion_matrix(y_test,np.argmax(prediction, axis = 1), labels=None, sample_weight=None))
print(sklearn.metrics.classification_report (y_test, np.argmax(prediction, axis = 1)))

