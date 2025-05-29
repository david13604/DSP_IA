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
    idx = np.random.randint(0, 2430)

    espectro = x_test[idx]
    peaks = y_test[idx]

    return espectro, peaks

def predecir_peaks(espectro):
    espectro = espectro.reshape(1, -1)  # Reshape para que sea compatible con la red
    prediccion = model.predict(espectro)
    return prediccion

if __name__ == "__main__":

    espectro, peaks = pick_random_sample(x_test, y_test)

    print("Shape of espectro:", espectro.shape)
    print("Shape of peaks:", peaks.shape)


    prediccion = predecir_peaks(espectro)


    print(prediccion.shape)
    print("Predicted shape:", prediccion[0].shape)
    #print(prediccion[0])

    for i in range(15):
        print(f"frecuancia dataset {peaks[i]} - frecuencias predichas {prediccion[0][i]}")

    print('\n'*2)

    for i in range(15,30):    
        print(f"amplitud dataset {peaks[i]} - amplitudes predichas {prediccion[0][i]}")
    #pred_peak_freqs = prediccion[0][:, 0]
    #pred_peak_amps = prediccion[0][:, 1]
    """
    plt.figure(figsize=(12, 6))
    plt.plot(espectro, label="Spectrum")
    plt.scatter(peak_freqs, peak_amps, color='red', label="Actual Peaks")
    plt.scatter(pred_peak_freqs, pred_peak_amps, color='green', label="Predicted Peaks", marker='x')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Spectrum with Actual and Predicted Peaks")
    plt.legend()
    plt.show()
    """