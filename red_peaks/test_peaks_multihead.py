import keras
import numpy as np
import matplotlib.pyplot as plt

# Cargar modelo multi-head
model = keras.models.load_model("peaks_model_multihead.h5")

# Cargar datos de prueba
data = np.load("dataset.npz")
x_test = data["X"]
y_test = data["Y"]  # Shape: (2431, 15, 2)

def pick_random_sample(x_test, y_test):
    espectro = x_test[1000] #1000 fue buen resultado, Shaepe: (131072,)
    peaks = y_test[1000]  # Shape: (15, 2)
    return espectro, peaks

def predecir_peaks(espectro):
    espectro = espectro.reshape(1, -1)  # Reshape para batch
    pred_amp, pred_freq = model.predict(espectro, verbose=0)
    return pred_amp.flatten(), pred_freq.flatten()

def plot_prediction_on_spectrum(mag, peak_freqs, peak_amps, fs=44100, title="Predicción sobre el espectro"):
    w = np.linspace(0, fs / 2, len(mag))  # eje de frecuencias en Hz
    peak_freqs_hz = peak_freqs * (fs / 2)  # Normalizado → Hz

    plt.figure(figsize=(10, 5))
    #plt.semilogx(w, mag, label="Espectro (dB)")
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

    print("Shape del espectro:", espectro.shape)

    pred_amps, pred_freqs = predecir_peaks(espectro[0:5000]) 

    print("valores reales")
    print("Amplitudes:", peaks[:, 1])
    print("Frecuencias (normalizadas):", peaks[:, 0])

    print("Predicciones:")
    print("Amplitudes:", pred_freqs)
    print("Frecuencias (normalizadas):", pred_amps)

    plot_prediction_on_spectrum(espectro, peaks[:,0], peaks[:,1], title="peaks reales sobre el espectro")
    plot_prediction_on_spectrum(espectro, pred_amps, pred_freqs, title="peaks inferidos sobre el espectro")
