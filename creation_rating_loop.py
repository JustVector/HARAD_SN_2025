import keras_manipulation as km
### Koniecznie Python 3.10 ( 3.8-3.11) inaczej tensorFlow odwala, przynajmniej w PyCharmie:)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score





#####config Section###############################
file_name:str = "updated_version.csv"

set_df(file_name=file_name)

