#In[]
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from sklearn.model_selection import train_test_split
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#In[]
data = pd.read_csv('/home/u108029050/m/data/cgmh_six_classes.csv')
data.head()
#In[]
print(f"There are {len(data)} rows in the dataset.")
#In[]
print(sum(data["y_true"].value_counts() == 1))
print(data["y_true"].nunique())
#In[]
# Filtering the rare terms.
data_filtered = data.groupby("y_true").filter(lambda x: len(x) > 1)
data_filtered.shape
#In[]
# data_filtered["y_true"] = data_filtered["y_true"].apply(lambda x: literal_eval(x))
# data_filtered["y_true"].values[:5]
# %%
test_split = 0.1
# Initial train and test split.
train_df, test_df = train_test_split(data_filtered,test_size=test_split,stratify=data_filtered["y_true"].values,)
# 將stratify=X就是按照X中的比例分配，將stratify=y就是按照y中的比例分配 
# Splitting the test set further into validation
# and new test sets.
val_df = test_df.sample(frac=0.5)#就是抽取其中50%
test_df.drop(val_df.index, inplace=True)#inplace=True：不創建新的對象，直接對原始對象進行修改；

print(f"Number of rows in training set: {len(train_df)}")
print(f"Number of rows in validation set: {len(val_df)}")
print(f"Number of rows in test set: {len(test_df)}")
# %%
train_df["X"].apply(lambda x: len(x.split(" "))).describe()
# %%
max_seqlen = 150
batch_size = 128
padding_token = "<pad>" #填充
auto = tf.data.AUTOTUNE # GPU訓練的同時CPU可以準備數據，從而提升訓練流程的效率

def make_dataset(dataframe, is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["X"].values, dataframe["y_true"].values)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)
# %%

train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)
# %%
vocabulary = set()
train_df["X"].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)
print(vocabulary_size)
# %%
text_vectorizer = layers.TextVectorization(
    max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
)

# `TextVectorization` layer needs to be adapted as per the vocabulary from our
# training set.
with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

train_dataset = train_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
validation_dataset = validation_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
test_dataset = test_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
# %%
def make_model():
    shallow_mlp_model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(vocabulary_size, activation="sigmoid"),
        ]  # More on why "sigmoid" has been used here in a moment.
    )
    return shallow_mlp_model
# %%
epochs = 20

shallow_mlp_model = make_model()
shallow_mlp_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]
)

history = shallow_mlp_model.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("loss")
plot_result("categorical_accuracy")
# %%
