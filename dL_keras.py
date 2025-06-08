### Koniecznie Python 3.10 ( 3.8-3.11) inaczej tensorFlow odwala, przynajmniej w PyCharmie:)
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


df = pd.read_csv("updated_version.csv")
print(df.head(3))

#Podział ramki danych na ramkę ze zmiennymi opisującymi i zmienną objaśnianą
x = df.iloc[:, 0:9]
y = df.iloc[:, 9]

# normalizacja danych do skali ~(mean = 0, s = 1)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# podział na sekcje uczące, walidacyjne i testowe
# najpierw 8:2 - uczący testowy (x_train = wejście na sieć, y_train = oczekiwane wyjście
# x_test - testowy input na wytrenowaną sieć, y_test = oczekiwany output z wytrenowanego modelu
x_train, x_, y_train, y_ = train_test_split(x_scaled, y, test_size=0.20, random_state=1)

# potem set testowy dzielimy na pół i mamy - 80% uczący, 10% walidacyjny, 10% testowy
x_val, x_test, y_val, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

print("x_train:", x_train.shape[0])
print("x_val:", x_val.shape[0])
print("x_test:", x_test.shape[0])


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


# Model 1 – ReLU
model_relu = Sequential([
    Dense(64, activation='relu'),    BatchNormalization(),
    Dense(32, activation='relu'),    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

# Model 2 – LeakyReLU - akceptuje wartości poniżej zera, nie "zabija neuronów"
model_leaky = Sequential([
    Dense(64),    BatchNormalization(),    LeakyReLU(negative_slope=0.01),
    Dense(32),    BatchNormalization(),    LeakyReLU(negative_slope=0.01),
    Dense(1, activation='sigmoid')
])


# Kompilacja
# a to sobie juz poczytajcie :P loss -> funkcja straty wg. której "karany i nagradzany" jest model.
for model in [model_relu, model_leaky]:
    model.compile(
        optimizer=AdamW(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision()]
    )


# funkcja hamulcowa - przerywa uczenie, jeśli nie poprawia sytuacji, chroni przed przeuczeniem.
# monitor - obserwowany parametr, min delta - minimalna zmiana obserwowanego param. żeby iteracja była uznana za sensowną.
# cierpliwość - liczba "bezsensownych" epok, zanim funkcja zatrzyma uczenie,
# restore.... - jak sama nazwa wskazuje ;-)
early_stop = EarlyStopping(monitor='val_loss', min_delta = 0.1, patience=10, restore_best_weights=True)



history_relu = model_relu.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

history_leaky = model_leaky.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

#obiekt model.evaluate() zwraca listę wartości wyników [funkcja straty, dokładność]
loss_relu, acc_relu, prec_relu = model_relu.evaluate(x_test, y_test, verbose=1)
loss_leaky, acc_leaky, prec_leaky = model_leaky.evaluate(x_test, y_test, verbose=1)

print(f"ReLU   - Test Accuracy: {acc_relu:.4f}, Loss: {loss_relu:.4f}, Prec: {prec_relu:.4f}")
print(f"LeakyR - Test Accuracy: {acc_leaky:.4f}, Loss: {loss_leaky:.4f}, Prec: {prec_leaky:.4f}")


##Wykresiki, bo wszyscy je kochamy.
def plot_history(history1, history2, label1='ReLU', label2='LeakyReLU'):
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

plot_history(history_relu, history_leaky)

# zapisuje pliki z modelem do późniejszego użycia.
#model.save("model_ReLu.keras")
#model.save("model_LeakReLu.keras")

#Mi się udało dojść do sporadycznych 93 % acc, średnio 92%, manipulując ilościa epok, cierpliwością i batchem.
#funkcję straty można by jeszcze rozważyć "focal_loss"
# Powodzenia ! :D

