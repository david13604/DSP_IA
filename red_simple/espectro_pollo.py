import librosa
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

def normalize_signal(signal, axis=None):
    # Normalize along the given axis (per-sample if axis is specified)
    min_val = tf.reduce_min(signal, axis=axis, keepdims=True)
    max_val = tf.reduce_max(signal, axis=axis, keepdims=True)
    return (signal - min_val) / tf.maximum(max_val - min_val, 1e-6)

path = "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/red_simple/Pollo_scream.mp3"

y, sr = librosa.load(path, sr=44100, mono=True)



Y = tf.signal.rfft(tf.cast(y, tf.float32))
mag = tf.math.abs(Y) + 1e-6
phase = tf.math.angle(Y)

norm_mag = normalize_signal(mag)
real_part = norm_mag * tf.cos(phase)
imag_part = norm_mag * tf.sin(phase)

# plot
plt.figure(figsize=(10, 4))
plt.plot(real_part)
plt.plot(imag_part)
plt.legend(["Real Part", "Imaginary Part"])
plt.title("FFT of Pollo Scream")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")
plt.grid()
plt.show()