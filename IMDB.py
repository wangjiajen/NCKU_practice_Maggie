#In[]
from tqdm import tqdm
# %%
import os

imdb_dir = '/home/u108029050/m/aclImdb' # Data directory
train_dir = os.path.join(imdb_dir, 'train') # Get the path of the train set

# Setup empty lists to fill
labels = []
texts = []

# First go through the negatives, then through the positives
for label_type in ['neg', 'pos']:
    # Get the sub path
    dir_name = os.path.join(train_dir, label_type)
    print('loading ',label_type)
    # Loop over all files in path
    for fname in tqdm(os.listdir(dir_name)):
        
        # Only consider text files
        if fname[-4:] == '.txt':
            # Read the text file and put it in the list
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            # Attach the corresponding label
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
# %%
len(labels), len(texts)

# %%
import numpy as np
np.mean(labels)
# %%
print('Label',labels[24002])
print(texts[24002])

print('Label',labels[1])
print(texts[1])
# %%
from keras.preprocessing.text import Tokenizer
import numpy as np

max_words = 10000 # We will only consider the 10K most used words in this dataset

tokenizer = Tokenizer(num_words=max_words) # Setup
tokenizer.fit_on_texts(texts) # Generate tokens by counting frequency
sequences = tokenizer.texts_to_sequences(texts) # Turn text into sequence of numbers
# %%
word_index = tokenizer.word_index
print('Token for "the"',word_index['the'])
print('Token for "Movie"',word_index['movie'])
print('Token for "generator"',word_index['generator'])

# %%
# Display the first 10 words of the sequence tokenized
sequences[24002][:10]
# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences
maxlen = 100 # Make all sequences 100 words long
data = pad_sequences(sequences, maxlen=maxlen)
print(data.shape) # We have 25K, 100 word sequences now
# %%
labels = np.asarray(labels)

# Shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

training_samples = 20000  # We will be training on 10K samples
validation_samples = 5000  # We will be validating on 10000 samples

# Split data
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
# %%
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

embedding_dim = 50

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# %%
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
# %%
