import keras
import os
from scipy.signal import stft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
from espectro_pollo import savitzky_golay

os.environ["KERAS_BACKEND"] = "tensorflow"

LARGO = 12407 * 2
FS = 44100

import tensorflow as tf

def savitzky_golay_tf(y, window_size, order):
    # Calculate polynomial coefficients (same as your NumPy version)
    # For simplicity, use fixed coefficients for order=3, window_size=51
    # You can generalize this if needed
    half_window = (window_size - 1) // 2
    # Generate convolution kernel (coefficients)
    # This is a placeholder for actual Savitzky-Golay coefficients
    # For production, precompute and hardcode the coefficients or use tf.numpy_function
    coeffs = tf.constant([1.0 / window_size] * window_size, dtype=tf.float32)
    coeffs = tf.reshape(coeffs, [window_size, 1, 1])

    # Pad the signal at both ends
    y = tf.reshape(y, [1, -1, 1])  # [batch, width, channels]
    y_padded = tf.pad(y, [[0, 0], [half_window, half_window], [0, 0]], mode="REFLECT")

    # Apply convolution
    y_smooth = tf.nn.conv1d(y_padded, coeffs, stride=1, padding="VALID")
    y_smooth = tf.squeeze(y_smooth, axis=[-1])
    return y_smooth

class FFTLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FFTLayer, self).__init__(**kwargs)

    def call(self, inputs, fs=FS, largo=LARGO):
        f_C, A, beta = inputs

        f_min = tf.constant(100.0, dtype=tf.float32)
        f_max = tf.constant(fs / 2.0, dtype=tf.float32)

        log_f_min = tf.math.log(f_min)
        log_f_max = tf.math.log(f_max)

        f_M = tf.constant(223, dtype=tf.float32)

        t = tf.range(largo, dtype=tf.float32) / tf.cast(fs, tf.float32)
        t = tf.reshape(t, (1, 1, -1))  # para broadcasting

        margin = tf.constant(0.95, dtype=tf.float32)  # keep away from hard edges
        n_carriers = tf.cast(tf.shape(A)[1], tf.float32)
        A = tf.nn.tanh(A) / tf.sqrt(n_carriers + 1e-8)

        s_c = (tf.tanh(f_C) * margin + 1.0) * 0.5  # in (0.025, 0.975)
        log_f_C = log_f_min + s_c * (log_f_max - log_f_min)
        f_C = tf.exp(log_f_C)

        sorted_indices = tf.argsort(f_C, axis=1)
        f_C = tf.gather(f_C, sorted_indices, batch_dims=1)
        A = tf.gather(A, sorted_indices, batch_dims=1)
        beta = tf.gather(beta, sorted_indices, batch_dims=1)

        f_C = tf.expand_dims(f_C, -1)
        beta = tf.expand_dims(beta, -1)
        A = tf.expand_dims(A, -1)

        # Constrain FM index
        beta = tf.nn.softplus(beta)
        beta = tf.clip_by_value(beta, 0.0, 10.0)

        # Map phase to [0, 2π]
        two_pi = tf.constant(2.0 * np.pi, dtype=tf.float32)

        mod = tf.sin(two_pi * f_M * t)
        fm_signal = tf.reduce_sum(
            A * tf.sin(two_pi * f_C * t + beta * mod),
            axis=1,
        )

        fft_result = tf.signal.rfft(tf.cast(fm_signal, tf.float32), fft_length=[largo])

        # Predict raw magnitude
        mag = tf.abs(fft_result) + 1e-9
        mag = tf.where(tf.math.is_finite(mag), mag, tf.zeros_like(mag))
        mag = savitzky_golay_tf(mag, 51, 3)
        return mag

class FM_red:
    def __init__(
        self,
        input_shape,
        output_shape,
    ) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.model = None

        self.build_model()

    def build_model(self):
        input_layer = keras.layers.Input(shape=self.input_shape)

        x = keras.layers.Flatten()(input_layer)
        x = keras.layers.Dense(512, activation="relu")(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(128, activation="relu")(x)


        # Ahora divido en sub capas de amplitud, frecuencia carrier e indice de modulacion
        f_c = keras.layers.Dense(
            self.output_shape // 3,
            activation=None,
            name="f_C",
            bias_initializer=keras.initializers.Constant(0.0),
        )(x)
        A = keras.layers.Dense(
            self.output_shape // 3,
            activation=None,
            name="A",
            kernel_initializer=keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer="zeros",
        )(x)
        beta = keras.layers.Dense(
            self.output_shape // 3,
            activation=None,
            name="beta",
            bias_initializer=keras.initializers.Constant(-2.0),  # softplus(-2) ≈ 0.13
        )(x)

        print(f"f_C shape: {f_c.shape}, beta shape: {beta.shape}, A shape: {A.shape}")

        # Output only magnitude spectrum
        output_layer = FFTLayer(name="fft_layer")([f_c, A, beta])
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    def compile(self, learning_rate=0.0001):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=2.0)
        self.model.compile(optimizer=optimizer, loss=mag_loss)

    def fit(self, x_train, y_train, epochs, batch_size=1, validation_data=None):
        callbacks = [
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1),
            keras.callbacks.EarlyStopping(monitor="loss", patience=50, restore_best_weights=True, verbose=1),
        ]
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )
        return history

    def save(self, path="modelo_fm.h5"):
        self.model.save(path)
        print(f"Model saved to {path}")

def mag_loss(y_true, y_pred):
    # L1
    return tf.reduce_mean(tf.square(y_true - y_pred))


if __name__ == "__main__":
    sr = FS
    output_shape = 30

    # Load data for training
    path = "dataset_single.npz"

    x_train = np.load(path)["X"]
    # Magnitude
    x_train = np.abs(x_train).astype(np.float32)
    x_train = x_train[..., np.newaxis]       
    x_train = np.expand_dims(x_train, axis=0)

    path_y = (
        "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/red_simple/Pollo_scream.mp3"
    )

    input_shape = x_train.shape[1:]
    print(f"x_train shape: {x_train.shape}")

    y, sr = librosa.load(path_y, sr=44100, mono=True)

    Y = tf.signal.rfft(tf.cast(y, tf.float32))
    mag = tf.math.abs(Y) + 1e-6

    smooth_mag = savitzky_golay(mag, 51, 3)
    y_train = tf.expand_dims(smooth_mag, axis=0)

    print(f"y_train shape (magnitude only): {y_train.shape}")

    model = FM_red(input_shape, output_shape)
    model.compile()

    history = model.fit(x_train, y_train, epochs=1000, batch_size=1)

    model.save("modelo_fm.h5")

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Predict once
    y_pred = model.model.predict(x_train, verbose=0)[0]
    freqs = np.fft.rfftfreq(LARGO, d=1.0/FS)
    y_train_np = tf.squeeze(y_train, axis=0).numpy()
    scale = np.max(y_train_np) + 1e-9
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, y_train_np/scale, label="|Y| (target)")
    plt.plot(freqs, y_pred/scale, label="|Y_pred|")
    plt.xlim(0, 10000)
    plt.grid(); plt.legend()
    plt.tight_layout()
    plt.show()
   