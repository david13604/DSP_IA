import keras
import os
from scipy.signal import stft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa

os.environ["KERAS_BACKEND"] = "tensorflow"

LARGO = 12407 * 2
FS = 44100


def normalize_signal(signal, axis=None):
    # Normalize along the given axis (per-sample if axis is specified)
    min_val = tf.reduce_min(signal, axis=axis, keepdims=True)
    max_val = tf.reduce_max(signal, axis=axis, keepdims=True)
    return (signal - min_val) / tf.maximum(max_val - min_val, 1e-6)


class FFTLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FFTLayer, self).__init__(**kwargs)

    def call(self, inputs, fs=FS, largo=LARGO):
        f_C, A, beta = inputs

        f_min = tf.constant(20.0, dtype=tf.float32)
        f_max = tf.constant(fs / 2.0, dtype=tf.float32)

        log_f_min = tf.math.log(f_min)
        log_f_max = tf.math.log(f_max)

        f_M = tf.constant(440, dtype=tf.float32)

        t = tf.range(largo, dtype=tf.float32) / tf.cast(fs, tf.float32)
        t = tf.reshape(t, (1, 1, -1))  # para broadcasting

        margin = tf.constant(0.95, dtype=tf.float32)  # keep away from hard edges
        s_c = (tf.tanh(f_C) * margin + 1.0) * 0.5
        A = tf.nn.sigmoid(A)  # 0..1 is fine for amplitude

        log_f_C = log_f_min + s_c * (log_f_max - log_f_min)

        f_C = tf.exp(log_f_C)

        f_C = tf.expand_dims(f_C, -1)
        f_M = tf.expand_dims(f_M, -1)
        beta = tf.expand_dims(beta, -1)
        A = tf.expand_dims(A, -1)

        mod = tf.sin(2 * tf.constant(np.pi, dtype=tf.float32) * f_M * t)
        fm_signal = tf.reduce_sum(
            A * tf.sin(2 * tf.constant(np.pi, dtype=tf.float32) * f_C * t + beta * mod),
            axis=1,
        )

        fft_result = tf.signal.rfft(tf.cast(fm_signal, tf.float32))

        mag = tf.abs(fft_result)
        global_max_mag = tf.reduce_max(mag)
        global_max_mag = tf.maximum(global_max_mag, 1e-9)

        denom = tf.sqrt(tf.reduce_mean(tf.square(mag), axis=-1, keepdims=True) + 1e-9)
        denom = tf.stop_gradient(denom)

        # Extract real & imag directly (avoid angle)
        real_norm = tf.math.real(fft_result) / global_max_mag
        imag_norm = tf.math.imag(fft_result) / global_max_mag

        out = tf.stack([real_norm, imag_norm], axis=-1)
        out = tf.where(tf.math.is_finite(out), out, tf.zeros_like(out))
        return out  # Solo la mitad positiva, 2 canales


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

        # Stronger downsampling to keep GRU tractable
        x = keras.layers.AveragePooling1D(pool_size=16, strides=16)(input_layer)
        #x = keras.layers.AveragePooling1D(pool_size=8, strides=8)(x)

        # Recurrent encoder over time
        x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True))(x)
        x = keras.layers.GaussianNoise(0.01)(x)  # optional stabilization
        x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=False))(x)

        # Small MLP head
        x = keras.layers.Dense(128, activation="relu")(x)

        # Ahora divido es sub capas de amplitud, frecuencia carrier e indice de modulacion
        f_c = keras.layers.Dense(
            self.output_shape // 3, activation=None, name="f_C"
        )(
            x
        )  # frecuencia carrier
        A = keras.layers.Dense(self.output_shape // 3, activation=None, name="A")(
            x
        )  # amplitud
        beta = keras.layers.Dense(
            self.output_shape // 3, activation=None, name="beta"
        )(x)

        print(f"f_C shape: {f_c.shape}, f_M shape: {beta.shape}, A shape: {A.shape}")
        # ahora junto todo porque me pide una salida
        output_layer = FFTLayer(name="fft_layer")([f_c, A, beta])
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    def compile(self, learning_rate=0.001):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
        self.model.compile(optimizer=optimizer, loss=complex)

    def fit(self, x_train, y_train, epochs, batch_size=32, validation_data=None):
        callbacks = [keras.callbacks.TerminateOnNaN()]
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
        )
        return history

    def save(self, path="modelo_fm.h5"):
        self.model.save(path)

        print(f"Model saved to {path}")


def complex(y_true, y_pred):
    real_diff = y_true[..., 0] - y_pred[..., 0]
    imag_diff = y_true[..., 1] - y_pred[..., 1]
    return tf.reduce_mean(tf.abs(real_diff)) + tf.reduce_mean(tf.abs(imag_diff)) + tf.reduce_mean(
        tf.abs(tf.complex(real_diff, imag_diff))
    )


def spectral_convergence_loss(y_true_complex, y_pred_complex):
    # Convert back to complex numbers to calculate magnitude
    y_true = tf.complex(y_true_complex[..., 0], y_true_complex[..., 1])
    y_pred = tf.complex(y_pred_complex[..., 0], y_pred_complex[..., 1])

    # Calculate magnitudes
    mag_true = tf.abs(y_true)
    mag_pred = tf.abs(y_pred)

    # Frobenius norm of the difference in magnitudes
    spectral_conv = tf.norm(mag_true - mag_pred, ord="euclidean", axis=-1) / (
        tf.norm(mag_true, ord="euclidean", axis=-1) + 1e-9
    )

    return tf.reduce_mean(spectral_conv)


def log_magnitude_loss(y_true_complex, y_pred_complex):
    y_true = tf.complex(y_true_complex[..., 0], y_true_complex[..., 1])
    y_pred = tf.complex(y_pred_complex[..., 0], y_pred_complex[..., 1])

    mag_true = tf.abs(y_true)
    mag_pred = tf.abs(y_pred)

    # Add a small epsilon to avoid log(0)
    log_mag_true = tf.math.log(mag_true + 1e-9)
    log_mag_pred = tf.math.log(mag_pred + 1e-9)

    return tf.reduce_mean(tf.abs(log_mag_true - log_mag_pred))


def combined_loss(y_true, y_pred):
    # You can weigh the two losses if needed
    alpha = 0.5
    sc_loss = spectral_convergence_loss(y_true, y_pred)
    lm_loss = log_magnitude_loss(y_true, y_pred)
    return alpha * sc_loss + (1.0 - alpha) * lm_loss


if __name__ == "__main__":
    sr = FS
    input_shape = (58797, 2)
    output_shape = 6

    # Load data for training
    path = "dataset_single.npz"

    x_train = np.load(path)["X"]
    # Reshape to match input shape
    x_train = np.stack([np.real(x_train), np.imag(x_train)], axis=-1).astype(np.float32)
    x_train = np.expand_dims(x_train, axis=0)

    print(f"x_train shape: {x_train.shape}")

    path_y = (
        "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/red_simple/Pollo_scream.mp3"
    )

    y, sr = librosa.load(path_y, sr=FS, mono=True)

    Y = tf.signal.rfft(tf.cast(y, tf.float32))

    mag = tf.abs(Y)
    global_max_mag = tf.reduce_max(mag)
    global_max_mag = tf.maximum(global_max_mag, 1e-9)

    real_norm = tf.math.real(Y) / global_max_mag
    imag_norm = tf.math.imag(Y) / global_max_mag

    y_stack = tf.stack([real_norm, imag_norm], axis=-1)

    # Igualar LARGO
    target_len = LARGO // 2 + 1
    y_stack = (
        y_stack[:target_len]
        if tf.shape(y_stack)[0] >= target_len
        else tf.pad(y_stack, [[0, target_len - tf.shape(y_stack)[0]], [0, 0]])
    )
    y_train = y_stack[tf.newaxis, ...].numpy().astype(np.float32)

    print(f"y_train shape: {y_train.shape}")

    model = FM_red(input_shape, output_shape)
    model.compile()

    history = model.fit(x_train, y_train, epochs=20, batch_size=1)

    model.save("modelo_fm.h5")

    # Plot history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(3, 1, 1)
    # Plot real and imaginary parts of the FFT
    # real
    y_pred = model.model.predict(x_train)[0]
    plt.plot(y_train[0, :, 0], label="Real Part")
    plt.plot(y_pred[:, 0], label="Predicted Real Part")
    plt.grid()
    plt.legend()
    # imaginary
    plt.subplot(3, 1, 2)
    plt.plot(y_train[0, :, 1], label="Imaginary Part")
    plt.plot(y_pred[:, 1], label="Predicted Imaginary Part")
    plt.legend()
    plt.grid()
    # magnitude
    magnitude = tf.sqrt(
        tf.square(y_train[0, :, 0]) + tf.square(y_train[0, :, 1])
    )
    predicted_magnitude = tf.sqrt(
        tf.square(y_pred[:, 0]) + tf.square(y_pred[:, 1])
    )
    plt.subplot(3, 1, 3)
    plt.plot(magnitude, label="Magnitude")
    plt.plot(predicted_magnitude, label="Predicted Magnitude")
    plt.legend()
    plt.grid()
    plt.show()
