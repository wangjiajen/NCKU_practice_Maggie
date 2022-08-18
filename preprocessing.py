import pandas as pd
from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def preprocess(max_words=10000):
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
    X_data = data[
        'summary']  # Combine title and description (better accuracy than using them as separate features)
    y_data = data['ClassIndex'].apply(
        lambda x: x - 1).values  # Class labels need to begin from 0
    x_testdata = testdata[
        'summary']  # Combine title and description (better accuracy than using them as separate features)
    y_testdata = testdata['ClassIndex'].apply(
        lambda x: x - 1).values  # Class labels need to begin from 0

    #Max Length of sentences in Train Dataset
    maxlen = X_data.map(lambda x: len(x.split())).max()
    data.head()

    # max_words = 10000 # 僅考慮資料集中的前10000個單詞
    # maxlen = 100 # 100個文字後切斷評論

    tok = Tokenizer(num_words=max_words)  # 實例化一個只考慮最常用10000詞的分詞器
    tok.fit_on_texts(X_data.values)  # 建構單詞索引

    # 將文字轉成整數list的序列資料
    X_data = tok.texts_to_sequences(X_data)
    x_testdata = tok.texts_to_sequences(x_testdata)

    X_data = keras.preprocessing.sequence.pad_sequences(X_data, maxlen=maxlen)
    x_testdata = keras.preprocessing.sequence.pad_sequences(x_testdata,
                                                            maxlen=maxlen)
    word_index = tok.word_index  #單詞和數字的字典

    training_samples = 96000  # We will be training on 10K samples
    validation_samples = 24000  # We will be validating on 10000 samples
    testing_samples = 7600
    # Split data
    X_train = X_data[:training_samples]
    y_train = y_data[:training_samples]
    X_val = X_data[training_samples:training_samples + validation_samples]
    y_val = y_data[training_samples:training_samples + validation_samples]
    X_test = x_testdata[:testing_samples]
    y_test = y_testdata[:testing_samples]
    # print("X_train set:", X_train.shape)
    # print("y_train set:", y_train.shape)
    # print("X_test set:", X_test.shape)
    # print("y_test set:", y_test.shape)
    # print("X_val set:", X_val.shape)
    # print("y_val set:", y_val.shape)

    return maxlen, word_index, X_train, y_train, X_val, y_val, testdata, X_test, y_test
