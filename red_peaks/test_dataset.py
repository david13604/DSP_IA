import numpy as np
import matplotlib.pyplot as plt

dataset = np.load('dataset.npz')
"""
Llaves:
X: (131072,) espectro a sr = 44100 Hz
Y: (15,2) contiene las frecuencias y las amplitudes de los peaks
Ambos estan completamente normalizados
"""

def pick_random_sample():

    print(dataset['X'].shape)
    espectro = dataset['X'][0]
    peaks = dataset['Y'][0]

    return espectro, peaks

def plot_spectrum_with_peaks(mag, peak_freqs, peak_amps, title="Spectrum with Peaks"):
    peak_freqs = peak_freqs * (44100 / 2) 
    w = np.linspace(0, 44100 / 2, len(mag)) 

    plt.figure(figsize=(10, 5))
    plt.plot(w, mag, label="Magnitud del espectro")
    plt.scatter(peak_freqs, peak_amps, color="red", label="Picos detectados", zorder=5)
    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__== "__main__":
    espectro, peaks = pick_random_sample()

    peak_freqs = peaks[:, 1] 
    peak_amps = peaks[:, 0]

    #peaks = peaks.reshape(1, 30)  # Reshape para que sea compatible con la red
    plot_spectrum_with_peaks(espectro, peak_amps, peak_freqs)