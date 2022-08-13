#In[]
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from sklearn.model_selection import train_test_split
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %%
arxiv_data = pd.read_csv("https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv")
arxiv_data.head()
# %%
print(f"There are {len(arxiv_data)} rows in the dataset.")
# %%
total_duplicate_titles = sum(arxiv_data["titles"].duplicated())
print(f"There are {total_duplicate_titles} duplicate titles.")
