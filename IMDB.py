#In[]
from tqdm import tqdm
# %%
import os

imdb_dir = '/home/u108029050/m/aclImdb' # Data directory
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

vocab_size = 10000 # We will only consider the 10K most used words in this dataset

tokenizer = Tokenizer(num_words=vocab_size) # Setup
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
indices = np.arange(data_test.shape[0])
np.random.shuffle(indices)
X_test = data_test[indices]
y_test = labels_test[indices]
indices = np.arange(data_train.shape[0])
np.random.shuffle(indices)
X_train = data_train[indices]
y_train = labels_train[indices]
# %%
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

embedding_dim = 100

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# %%
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history=model.fit(X_train,y_train, batch_size=16, epochs=10, validation_split=0.2)
# In[]
history.history
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# In[]
prediction = model.predict(X_test)
# In[]
import sklearn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from tensorflow.keras.utils import plot_model

print(sklearn.metrics.confusion_matrix(y_test,np.argmax(prediction, axis = 1), labels=None, sample_weight=None))
print(sklearn.metrics.classification_report (y_test, np.argmax(prediction, axis = 1)))
