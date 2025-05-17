import librosa
import scipy
import numpy as np
import matplotlib.pyplot as plt

rate = 44100
n_peaks = 15 # Number of peaks to show

aud_data, sr = librosa.load("Pollo_scream.mp3",sr=rate, mono=True)

len_data = len(aud_data)

# Padding
channel_1 = np.zeros(2**(int(np.ceil(np.log2(len_data)))))
channel_1[0:len_data] = aud_data

fourier = np.fft.fft(channel_1)

w = np.linspace(0, 44000, len(fourier))

fourier_to_plot = fourier[0:len(fourier)//2]
w = w[0:len(fourier)//2]

normalized_fourier = fourier_to_plot / np.max(np.abs(fourier_to_plot))
peaks, _ = scipy.signal.find_peaks(20 * np.log10(np.abs(normalized_fourier)), distance=75, prominence=6)
print("len(peaks):", len(peaks))

#Peak index and value
peak_tuple = [(20 * np.log10(np.abs(normalized_fourier[peak])), peak) for peak in peaks]
peak_tuple.sort(reverse=True)

peaks = [peak[1] for peak in peak_tuple[:n_peaks]]

plt.figure(1)

plt.plot(w, 20 * np.log10(np.abs(normalized_fourier)), label='Real')
plt.plot(w[peaks], 20 * np.log10(np.abs(normalized_fourier[peaks])), "x", label='Peaks')
plt.axis([0, 7500, -80,80])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.show()