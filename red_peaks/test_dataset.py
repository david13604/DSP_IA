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
    idx = np.random.randint(0, 2430)

    print(dataset['X'].shape)
    espectro = dataset['X'][idx]
    peaks = dataset['Y'][idx]

    return espectro, peaks

def plot_spectrum_with_peaks(mag, peak_freqs, peak_amps, title="Spectrum with Peaks"):
    plt.figure(figsize=(12, 6))
    plt.plot(mag, label="Spectrum")
    plt.scatter(peak_freqs, peak_amps, color='red', label="Peaks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.legend()
    plt.show()

if __name__== "__main__":
    espectro, peaks = pick_random_sample()

    peak_freqs = peaks[:, 0] 
    peak_amps = peaks[:, 1]

    print(peak_freqs)
    print(peak_amps)

    #plot_spectrum_with_peaks(espectro, peak_amps, peak_freqs)