import keras
import os
from scipy.signal import stft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
from espectro_pollo import savitzky_golay
from entrenar_magnitud import FFTLayer

LARGO = 12407 * 2
FS = 44100

path = "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/red_simple/dataset_single.npz"

x_train = np.load(path)["X"]
# Magnitude
x_train = np.abs(x_train).astype(np.float32)
x_train = x_train[..., np.newaxis]
x_train = np.expand_dims(x_train, axis=0)

path_y = "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/red_simple/Pollo_scream.mp3"

y, sr = librosa.load(path_y, sr=44100, mono=True)

Y = tf.signal.rfft(tf.cast(y, tf.float32))
mag = tf.math.abs(Y) + 1e-6

smooth_mag = savitzky_golay(mag, 51, 3)
y_train = tf.expand_dims(smooth_mag, axis=0)

print(f"y_train shape (magnitude only): {y_train.shape}")

model = keras.models.load_model(
    "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/red_simple/modelo_fm.h5",
    custom_objects={"FFTLayer": FFTLayer},
    compile=False,
)

# Predict once
y_pred = model.predict(x_train, verbose=0)[0]
freqs = np.fft.rfftfreq(LARGO, d=1.0 / FS)
y_train_np = tf.squeeze(y_train, axis=0).numpy()
scale = np.max(y_train_np) + 1e-9
plt.figure(figsize=(10, 4))
plt.plot(freqs, y_train_np / scale, label="|Y| (target)")
plt.plot(freqs, y_pred / scale, label="|Y_pred|")
plt.xlim(0, 10000)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
