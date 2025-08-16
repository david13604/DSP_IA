import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf


# Load the autoencoder
model = keras.models.load_model(
    "primero_autoencoder.keras", compile=False
)

# Load a sample from your dataset
data = np.load("dataset_single.npz")
X = data["X"]
X = np.expand_dims(X, axis=-1).astype(np.float32)

# Select one sample
n = 1  # sample number change this to test different samples
input_sample = X[n : n + 1]  # random stft

# Predict output
output_sample = model.predict(input_sample)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(output_sample), aspect="auto", origin="lower", cmap="viridis")
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
