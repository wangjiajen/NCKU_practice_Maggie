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

vocab_size, word_index, X_train, y_train, X_val, y_val, testdata, X_test, y_test = preprocess(
)

from transformer import TransformerBlock, TokenAndPositionEmbedding
#In[]
# dataset = "agnews"
# assert dataset in ["agnews", "imdb"]
from keras.models import load_model
from keras_bert import get_custom_objects

model_path = "transformer_agnews_wiki_em.best.h5"
model = load_model(model_path,
                   custom_objects={
                       'TransformerBlock': TransformerBlock,
                       'TokenAndPositionEmbedding': TokenAndPositionEmbedding
                   })

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
