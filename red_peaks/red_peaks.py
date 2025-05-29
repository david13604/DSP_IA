import keras
from keras.losses import MeanSquaredError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#input: sinusoide cruda
# nosotros: probemos pasarle la sinusoide o pasarle fft o pasarle la stft -> parametros FM

def build_model(input_shape): 
    model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(30, activation="sigmoid")
])
    model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["accuracy"])
    return model

def train_model(model, x_train, y_train, epochs=10, batch_size=32):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

if __name__ == "__main__":
 
    df = np.load("dataset.npz") #cargar datos

    x_train = df["X"] #Espectros de frecuencias
    y_train = df["Y"] #Peaks de frecuencias

    y_train = y_train.reshape(2431, 30) #Reshape para que sea compatible con la red

    input_shape = (x_train.shape[1],)
    print(input_shape)

    model = build_model(input_shape) 
    model.summary() 

    history = train_model(model, x_train, y_train, epochs=100) #Entrenar el modelo

    # Graficar la pérdida de entrenamiento y validación
    plt.plot(history.history["loss"], label="Pérdida de entrenamiento")
    plt.plot(history.history["val_loss"], label="Pérdida de validación")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Curvas de pérdida")
    plt.show()

    model.save("peaks_model.h5") #guardo