import keras
import pandas as pd
import numpy as np
from sin_losses import sin_error_fcn
import matplotlib.pyplot as plt

#input: sinusoide cruda
# nosotros: probemos pasarle la sinusoide o pasarle fft o pasarle la stft -> parametros FM

def build_model(input_shape, loss_fcn): 
    model = keras.Sequential(
        [
            keras.layers.Dense(16, activation="relu", input_shape=input_shape),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(2, activation="sigmoid")
        ]  
    )
    model.compile(optimizer="adam", loss=loss_fcn, metrics=["accuracy"])
    return model

def train_model(model, x_train, epochs=10, batch_size=32):
    history = model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

if __name__ == "__main__":
 
    df = pd.read_csv("sine_data.csv") #cargar datos

    x_train = df.values.astype(np.float32)

    input_shape = (x_train.shape[1],)
    print(input_shape)

    model = build_model(input_shape, sin_error_fcn) #construyo modelo

    history = train_model(model, x_train, epochs=100) #entreno

    # Graficar la pérdida de entrenamiento y validación
    plt.plot(history.history["loss"], label="Pérdida de entrenamiento")
    plt.plot(history.history["val_loss"], label="Pérdida de validación")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Curvas de pérdida")
    plt.show()
    
    model.save("sine_model.h5") #guardo
