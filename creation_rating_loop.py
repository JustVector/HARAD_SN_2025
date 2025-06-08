import keras_manipulation as km
### Koniecznie Python 3.10 ( 3.8-3.11) inaczej tensorFlow odwala, przynajmniej w PyCharmie:)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


#####config Section###############################
# file_name:str = "C:\\Users\\loern\\eszi\\nowy\\HARAD_SN_2025\\updated_version.csv"
file_name:str = "updated_version.csv"
layer_qty_list = [1]#,2,4,8] #do oceny
neurons_qty_combinations:list = [[2, 4, 8, 16, 32, 16, 8, 4], #Do oceny
                        #    [8, 8, 16, 16, 8, 8, 16, 16],
                        #    [128, 64, 32, 16, 8, 4, 2, 1],
                        #    [16, 8, 32, 2, 8, 16, 4, 8]
                           ]
# Compiling
optimizers_list:list = ["AdamW"]#, "SGD"] #do oceny
learning_rates_list:list = [0.001]#, 0.0005]#, 0.1] # do oceny
loss_fun_name_list:list = ["binary_crossentropy"]#, "categorical_crossentropy"]#], "hinge"] # do oceny
momentum_val:float = 0.9 # [0 - 1]

#early stop
min_delta:float = 0.01
patience_list:list = [15]

#fitting
epochs_list:list = [50]#, 1]
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



#####modeling Section #############################

#[x_train, x_val, x_test, y_train, y_val, y_test]
data_set:list = km.set_df(file_name=file_name)

for idx, row in the_df.iterrows():
    modele = km.set_keras_models_list(row["layer_qty"], row["neurons_qty_set"])
    km.compile_model_seq(modele, row["optimizer"], row["learning_rate"], row["loss_function"], momentum_val)
    EarlyStop = km.set_early_stop_function(min_delta, patience=row["patience"])
    history = km.train_model(modele, data_set, EarlyStop, row["epochs"], row["batch_size"], verbose=verbose, do_save=False)
    test_acc, test_loss, test_prec = km.test_model(modele, data_set[2], data_set[5])

    the_df.at[idx, "modele"] = modele
    the_df.at[idx, "history"] = history
    the_df.at[idx, "test_accuracy"] = test_acc
    the_df.at[idx, "test_loss"] = test_loss
    the_df.at[idx, "test_precision"] = test_prec
    the_df.loc[idx, "max_test_accuracy"] = max(test_acc)
    the_df.loc[idx, "best_model"] = modele[test_acc.index(max(test_acc))]
    the_df.loc[idx, "best_index"] = test_acc.index(max(test_acc))

hyperparameter_columns = [
    'layer_qty',
    'optimizer',
    'learning_rate',
    'loss_function',
    'patience',
    'epochs',
    'batch_size'
]

best_models_by_attribute = {}

for col in hyperparameter_columns:
    best_rows = the_df.loc[the_df.groupby(col)['max_test_accuracy'].idxmax()]
    print(best_rows)

    best_models_by_attribute[col] = best_rows
    for index, row in best_rows.iterrows():
        print(f"  {col}: {row[col]} -> Best Accuracy: {row['max_test_accuracy']:.4f}")

km.read_model_history(best_models_by_attribute)