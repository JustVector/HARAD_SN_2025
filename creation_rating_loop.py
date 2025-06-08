from logging import raiseExceptions

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

layer_qty_list = [1,2,4,8]
neurons_qty_combinations:list = [[2, 4, 8, 16, 32, 16, 8, 4],
                           [8, 8, 16, 16, 8, 8, 16, 16],
                           [128, 64, 32, 16, 8, 4, 2, 1],
                           [16, 8, 32, 2, 8, 16, 4, 8]
                           ]
# Compiling
optimizers_list:list = ["AdamW", "SGD"]
learning_rates_list:list = [0.001, 0.0005, 0.0001,0.1]
loss_fun_name_list:list = ["binary_crossentropy", "categorical_crossentropy", "hinge"]
momentum_val:float = 0.9 # [0 - 1]

#early stop
min_delta:float = 0.01
patience_list:list = [15]

#fitting
epochs_list:list = [1, 5]
batch_size_list:list = [32]
verbose:int = 0 #0,1,2

######loop Section#################################

the_df = km.build_config_df(
        layer_qty_list,
        neurons_qty_combinations,
        optimizers_list,
        learning_rates_list,
        loss_fun_name_list,
        momentum_val,
        min_delta,
        patience_list,
        epochs_list,
        batch_size_list
)

    #     "optimizer",
    #     "learning_rate",
    #     "loss_function",
    #     "patience",
    #     "epochs",
    #     "batch_size"
    #     df["momentum"] = momentum_val
    #     df["min_delta"] = min_delta
    #
    # # Kolumny na wyniki i obiekty

    #     df["modele"] = None
    #     df["history"] = None
    #     df["test_accuracy"] = None
    #     df["test_loss"] = None
    #     df["test_precision"] = None

#####modeling Section #############################

#[x_train, x_val, x_test, y_train, y_val, y_test]
data_set:list = km.set_df(file_name=file_name)


for idx, row in the_df.iterrows():
    # zamodeluj sieć,
    row["modele"] = km.set_keras_models_list(row["layer_qty"], row["neurons_qty_set"])

    # kompiluj modele,
    km.compile_model_seq(row["modele"], row["optimizer"], row["learning_rate"], row["loss_function"], momentum_val)

    # przygotuj funkcję early_stop
    EarlyStop = km.set_early_stop_function(min_delta, patience=row["patience"])

    #Naucz modele
    row["history"] = km.train_model(row["modele"], data_set, EarlyStop, row["epochs"], row["batch_size"], verbose=verbose, do_save=False)

    # testuj modele
    row["test_accuracy"], row["test_loss"], row["test_precision"] = km.test_model(row["modele"],data_set[2], data_set[5])

    # porównaj dwa
    km.plot_history_comparison(row["history"][1],row["history"][0])

