import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = keras.models.load_model("peaks_model.h5")

x_test = np.load("dataset.npz")["X"]
y_test = np.load("dataset.npz")["Y"]

y_test = y_test.reshape(2431, 30)  

def pick_random_sample(x_test, y_test):

    espectro = x_test[0]
    peaks = y_test[0]

    return espectro, peaks

def predecir_peaks(espectro):
    espectro = espectro.reshape(1, -1)  # Reshape para que sea compatible con la red
    prediccion = model.predict(espectro)
    return prediccion

def plot_prediction_on_spectrum(mag, peak_freqs, peak_amps, fs=44100, title="Predicción sobre el espectro"):
    w = np.linspace(0, fs / 2, len(mag))  # eje de frecuencias del espectro

    peak_freqs_hz = peak_freqs * (fs / 2)  # convertir de [0, 1] → Hz

    plt.figure(figsize=(10, 5))
    plt.plot(w, mag, label="Espectro")
    plt.scatter(peak_freqs_hz, peak_amps, color="red", label="Predicción (picos)", zorder=5)
    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    espectro, peaks = pick_random_sample(x_test, y_test)

    print("Shape of espectro:", espectro.shape)
    print("Shape of peaks:", peaks.shape)


    prediccion = predecir_peaks(espectro)

    prediccion = np.array(prediccion).flatten()  # Aplanar la predicción para facilitar el manejo


    print(prediccion.shape)
    print("Predicted shape:", prediccion[0].shape)

    peak_freqs = prediccion[::2]  # índices pares → frecuencias
    peak_amps = prediccion[1::2] 

    plot_prediction_on_spectrum(espectro, peak_freqs, peak_amps, title="Predicción de picos sobre el espectro")