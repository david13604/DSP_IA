import keras
import scipy
from scipy.signal import stft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
from entrenar_magnitud import FFTLayer
from espectro_pollo import savitzky_golay


def peak_envelope(mag, freqs):
    # Find peaks
    peaks, _ = scipy.signal.find_peaks(mag)
    # Interpolate envelope through peaks
    envelope_interp = scipy.interpolate.interp1d(
        freqs[peaks], mag[peaks], kind="cubic", fill_value="extrapolate"
    )
    return envelope_interp(freqs)


LARGO = 12407 * 2
FS = 44100

window_size = 51
order = 3

path_y = "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/red_simple/Pollo_scream.mp3"

y, sr = librosa.load(path_y, sr=44100, mono=True)

Y = tf.signal.rfft(tf.cast(y, tf.float32))
mag = tf.math.abs(Y)

smooth_mag = tf.abs(peak_envelope(mag, np.fft.rfftfreq(len(y), d=1.0/FS)))
y_train = tf.expand_dims(smooth_mag, axis=0)

print(f"y_train shape (magnitude only): {y_train.shape}")

model = keras.models.load_model(
    "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/red_simple/modelo_fm.h5",
    custom_objects={"FFTLayer": FFTLayer},
    compile=False,
)

# Predict once
y_pred = model.predict(y_train, verbose=0)[0]
freqs = np.fft.rfftfreq(LARGO, d=1.0 / FS)
y_train_np = tf.squeeze(y_train, axis=0).numpy()
scale = np.max(y_train_np)
plt.figure(figsize=(10, 4))
plt.plot(freqs, y_train_np / scale, label="|Y| (target)")
plt.plot(freqs, y_pred / scale, label="|Y_pred|")
plt.plot(freqs, tf.abs(peak_envelope(y_pred, freqs)) / scale, label="|Y_pred smooth|")
plt.xlim(0, 5000)
plt.ylim(0, 1.5)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
