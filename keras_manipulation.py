### Koniecznie Python 3.10 ( 3.8-3.11) inaczej tensorFlow odwala, przynajmniej w PyCharmie:)
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

########## model#############
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




def set_df(file_name: str, test_sample_part_ratio: float = 0.2, validation_sample_part_ratio: float = 0.5):
    """
    , target_var:str="" do zrobienia definicja zmiennej objaśnianej
    :param file_name:
    :param test_sample_part_ratio:
    :param validation_sample_part_ratio:
    :return:
    """
    df = pd.read_csv(file_name)

    # Podział ramki danych na ramkę ze zmiennymi opisującymi i zmienną objaśnianą
    x = df.iloc[:, 0:9]
    y = df.iloc[:, 9]

    # normalizacja danych do skali ~(mean = 0, s = 1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # podział na sekcje uczące, walidacyjne i testowe
    # najpierw 8:2 - uczący testowy (x_train = wejście na sieć, y_train = oczekiwane wyjście
    # x_test - testowy input na wytrenowaną sieć, y_test = oczekiwany output z wytrenowanego modelu
    x_train, x_, y_train, y_ = train_test_split(x_scaled, y, test_size=test_sample_part_ratio, random_state=1)

    # potem set testowy dzielimy na pół i mamy - 80% uczący, 10% walidacyjny, 10% testowy
    x_val, x_test, y_val, y_test = train_test_split(x_, y_, test_size=validation_sample_part_ratio, random_state=1)

    print("x_train:", x_train.shape[0])
    print("x_val:", x_val.shape[0])
    print("x_test:", x_test.shape[0])

    return [x_train, x_val, x_test, y_train, y_val, y_test]

def set_keras_models_list(layers_qty:int = 3, neurons_qty_combination:list = [64,32]):
    """
    Generuje modele wg. definicji:
    pozycja na liście kompilacji, funkcja aktywacji, BatchNormalization() + funkcja na wyjściu

    Przyjmuje zmienne:
    :param layers_qty: ilość warstw w modelu bez warstwy wyjściowej, int
    :param neurons_qty_combination:
    :return:


    parametry zmieniane na sztywno między modelami:
    funkcje aktywacji:

    funkcje aktywacji wyjścia:

    BatchNormalization:
    """
    compilation_list: list = []

    # Model 0 – ReLU + BatchNormalization + sigmoid out.
    layer_list = [
        layer
        for i in range(layers_qty)
        for layer in (
            Dense(neurons_qty_combination[i], activation='relu'),
            BatchNormalization()
        )
    ]
    # Definicja wyjścia
    output_layer: Dense = Dense(1, activation='sigmoid')
    layer_list.append(output_layer)
    # Generowanie obiektu
    model_relu = Sequential(layer_list)
    # append na return funkcji
    compilation_list.append(model_relu)


    # Model 1 – LeakyReLU + BatchNormalization + sigmoid out.
    layer_list = [
        layer
        for i in range(layers_qty)
        for layer in (
            Dense(neurons_qty_combination[i]), BatchNormalization(), LeakyReLU(negative_slope=0.01)
        )
    ]
    # Definicja wyjścia
    output_layer: Dense = Dense(1, activation='sigmoid')
    layer_list.append(output_layer)
    # Generowanie obiektu
    model_leaky = Sequential(layer_list)
    # append na return funkcji
    compilation_list.append(model_relu)


    # Model 2 – tanh + BatchNormalization + sigmoid out.
    layer_list = [
        layer
        for i in range(layers_qty)
        for layer in (
            Dense(neurons_qty_combination[i],activation='tanh'), BatchNormalization()
        )
    ]
    # Definicja wyjścia
    output_layer: Dense = Dense(1, activation='sigmoid')
    layer_list.append(output_layer)
    # Generowanie obiektu
    model_tanh = Sequential(layer_list)
    # append na return funkcji
    compilation_list.append(model_relu)


    # Model 3 – tanh + NO_BatchNormalization + sigmoid out.
    layer_list = [
        layer
        for i in range(layers_qty)
        for layer in (
            Dense(neurons_qty_combination[i], activation='tanh'), BatchNormalization()
        )
    ]
    # Definicja wyjścia
    output_layer: Dense = Dense(1, activation='sigmoid')
    layer_list.append(output_layer)
    # Generowanie obiektu
    model_tanh = Sequential(layer_list)
    # append na return funkcji
    compilation_list.append(model_relu)


    # Model 4 – ReLu + NO_BatchNormalization + sigmoid out.
    layer_list = [
        layer
        for i in range(layers_qty)
        for layer in (
            Dense(neurons_qty_combination[i], activation='relu')
        )
    ]
    # Definicja wyjścia
    output_layer: Dense = Dense(1, activation='sigmoid')
    layer_list.append(output_layer)
    # Generowanie obiektu
    model_relu_noBatchNorm = Sequential(layer_list)
    # append na return funkcji
    compilation_list.append(model_relu)


    # Model 5 – ReLu + BatchNormalization + Hard sigmoid out.
    layer_list = [
        layer
        for i in range(layers_qty)
        for layer in (
            Dense(neurons_qty_combination[i], activation='relu'), BatchNormalization()
        )
    ]
    # Definicja wyjścia
    output_layer: Dense = Dense(1, activation='hard_sigmoid')
    layer_list.append(output_layer)
    # Generowanie obiektu
    model_relu_hardSigmoid = Sequential(layer_list)
    # append na return funkcji
    compilation_list.append(model_relu)


    # Model 6 – ReLu + BatchNormalization + Hard sigmoid out.
    layer_list = [
        layer
        for i in range(layers_qty)
        for layer in (
            Dense(neurons_qty_combination[i], activation='relu'), BatchNormalization()
        )
    ]
    # Definicja wyjścia
    output_layer: Dense = Dense(1, activation='hard_sigmoid')
    layer_list.append(output_layer)
    # Generowanie obiektu
    model_relu_hardSigmoid = Sequential(layer_list)
    # append na return funkcji
    compilation_list.append(model_relu)


    # Model 7 – LeakyReLU + BatchNormalization + Hard sigmoid out.
    layer_list = [
        layer
        for i in range(layers_qty)
        for layer in (
            Dense(neurons_qty_combination[i]), BatchNormalization(), LeakyReLU(negative_slope=0.01)
        )
    ]
    # Definicja wyjścia
    output_layer: Dense = Dense(1, activation='hard_sigmoid')
    layer_list.append(output_layer)
    # Generowanie obiektu
    model_leakyRelu_hardSigmoid = Sequential(layer_list)
    # append na return funkcji
    compilation_list.append(model_relu)

    # Model 8 – Realy LeakyReLU + BatchNormalization + Hard sigmoid out.
    layer_list = [
        layer
        for i in range(layers_qty)
        for layer in (
            Dense(neurons_qty_combination[i]), BatchNormalization(), LeakyReLU(negative_slope=0.1)
        )
    ]
    # Definicja wyjścia
    output_layer: Dense = Dense(1, activation='hard_sigmoid')
    layer_list.append(output_layer)
    # Generowanie obiektu
    model_leakyRelu_hardSigmoid = Sequential(layer_list)
    # append na return funkcji
    compilation_list.append(model_relu)


    # Model 9 – Realy LeakyReLU + BatchNormalization + sigmoid out.
    input_layer_set = [Dense(64, activation='relu', input_shape=(10,))]
    layer_list = [
        layer
        for i in range(layers_qty)
        for layer in (
            Dense(neurons_qty_combination[i]), BatchNormalization(), LeakyReLU(negative_slope=0.1)
        )
    ]
    # Definicja wyjścia
    output_layer: Dense = Dense(1, activation='sigmoid')
    layer_list.append(output_layer)
    # Generowanie obiektu
    model_leakyRelu_hardSigmoid = Sequential(layer_list)
    # append na return funkcji
    compilation_list.append(model_relu)

###################################################################################################################
#Modele z warstwami innymi niż dense() ?


    return compilation_list


def compile_model_seq(compilation_list:list,
                      optimizer_name:str = "AdamW",
                      learning_rate_val:float = 0.001,
                      loss_fun_name:str = "binary_crossentropy",
                      momentum_val:float = 0.9
                      ):
    """
    kompiluje modele z zadanej listy wg. wskazanych parametrów
    :param compilation_list: list from set_keras_models_list()
    :param optimazer_name: AdamW, Adam, SGD
    :param learning_rate:
    :param loss_fun_name: 'binary_crossentropy', categorical_crossentropy, hinge...
    :return:
    """

    if optimizer_name == "AdamW":
        optimizer_formula:str = optimizer_name + f"(learning_rate={learning_rate_val})"
    elif optimizer_name == "Adam":
        optimizer_formula = optimizer_name + f"(learning_rate={learning_rate_val})"
    elif optimizer_name == "SGD":
        optimizer_formula = optimizer_name + f"(learning_rate={learning_rate_val}, momentum={momentum_val}, nesterov=True)"
    else:
        optimizer_formula = optimizer_name + f"(learning_rate={learning_rate_val})"


    for model in compilation_list:
        model.compile(
            optimizer=eval(optimizer_formula),
            loss=loss_fun_name,
            metrics=['accuracy', Precision()]
        )

    return


def set_early_stop_function(min_delta:float=0.1, patience:int=10):
    """
    zwraca postać funkcji "hamulcowej"
    :param min_delta: zakres [0.0-1.0]
    :param patience:
    :return:early_stop
    """
    early_stop = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, restore_best_weights=True)
    return early_stop

def train_model(model, data_list:list,
                early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10, restore_best_weights=True),
                epochs:int = 50,
                batch_size:int = 32,
                verbose=2,
                do_save:bool = False,):
    """

    :param model:
    :param data_list:
    :param early_stop:
    :param epochs:
    :param batch_size:
    :param verbose:
    :return: history - obiekt zawierający dane z uczenia modelu.
    """

    x_train, x_val, x_test, y_train, y_val, y_test = data_list
    pass
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stop],
                        verbose=verbose
                        )
    if do_save:
        model.save(f"{model}.keras")

    return history


def plot_history_comparison(history1, history2, label1='ReLU', label2='LeakyReLU'):
    plt.figure(figsize=(14,5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['val_accuracy'], label=f'{label1} - val acc')
    plt.plot(history2.history['val_accuracy'], label=f'{label2} - val acc')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history1.history['val_loss'], label=f'{label1} - val loss')
    plt.plot(history2.history['val_loss'], label=f'{label2} - val loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


