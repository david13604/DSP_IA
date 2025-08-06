import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from red import STFTLayer


# Load the trained model
model = keras.models.load_model(
    "modelo_fm.h5", compile=False, custom_objects={"STFTLayer": STFTLayer}
)

# Load a sample from your dataset
data = np.load("dataset_single.npz")
X = data["X"]
X = np.expand_dims(X, axis=-1).astype(np.float32)

# Select one sample
n = 0  # sample number change this to test different samples
input_sample = X[n : n + 1]  # random stft

# Predict output
output_sample = model.predict(input_sample)

# construct outputwave
fs = 44100
cant_muestras = int(300 * 44100 // 1000)
numero_fm = 84 // 4
fm_final = np.zeros(cant_muestras)
for i in range(0, numero_fm):
    f_C = output_sample[0, :, 0 + i * 4]
    I = output_sample[0, :, 1 + i * 4]
    A = output_sample[0, :, 2 + i * 4]
    f_M = output_sample[0, :, 3 + i * 4]
    # Generate time vector
    t = tf.linspace(0.0, cant_muestras / fs, cant_muestras)
    f_M = tf.expand_dims(f_M, axis=1)
    f_C = tf.expand_dims(f_C, axis=1)
    I = tf.expand_dims(I, axis=1)
    A = tf.expand_dims(A, axis=1)
    t = tf.expand_dims(t, axis=0)  
    mod = tf.sin(2 * np.pi * f_M * t)
    fm_signal = A * tf.sin(2 * np.pi * f_C * t + I * mod)
    fm_final += tf.reduce_sum(fm_signal, axis=0).numpy()
# STFT
stft_result = tf.signal.stft(fm_final, frame_length=128, frame_step=32, fft_length=128)
magnitude = tf.abs(stft_result)
log_magnitude = (
    20
    * tf.math.log(tf.maximum(magnitude, 1e-6))
    / tf.math.log(tf.constant(10.0, dtype=tf.float64))
)
min_val = tf.reduce_min(log_magnitude, axis=[0, 1], keepdims=True)
max_val = tf.reduce_max(log_magnitude, axis=[0, 1], keepdims=True)
norm_mag = (log_magnitude - min_val) / tf.maximum(max_val - min_val, 1e-6)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(norm_mag.numpy(), aspect="auto", origin="lower", cmap="viridis")
plt.title("Output")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(input_sample), aspect="auto", origin="lower", cmap="viridis")
plt.title("Input")
plt.colorbar()
plt.tight_layout()
plt.show()

# Calculate mse
mse = tf.reduce_mean((norm_mag - np.squeeze(input_sample)) ** 2).numpy()
print(f"MSE between input and output: {mse}")