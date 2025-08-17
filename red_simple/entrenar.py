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


class FFTLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FFTLayer, self).__init__(**kwargs)

    def call(self, inputs, fs=FS, largo=LARGO):
        f_C, A, f_M = inputs

        I = tf.constant(440 / FS, dtype=tf.float32)

        t = tf.range(largo, dtype=tf.float32) / float(fs)
        t = tf.reshape(t, (1, 1, -1))  # para broadcasting

        f_C = tf.expand_dims(f_C, -1)
        f_M = tf.expand_dims(f_M, -1)
        I = tf.expand_dims(I, -1)
        A = tf.expand_dims(A, -1)

        mod = tf.sin(2 * np.pi * f_M * t)
        fm_signal = tf.reduce_sum(A * tf.sin(2 * np.pi * f_C * t + I * mod), axis=1)

        fft_result = tf.signal.rfft(tf.cast(fm_signal, tf.float32))

        # # Compute magnitude and phase
        # magnitude = tf.math.abs(fft_result) + 1e-6
        # phase = tf.math.angle(fft_result)

        # # Normalizar log-magnitude
        # log_magnitude = (
        #     20 * tf.math.log(tf.maximum(magnitude, 1e-6)) / tf.math.log(10.0)
        # )
        # min_val = tf.reduce_min(log_magnitude, axis=[1], keepdims=True)
        # max_val = tf.reduce_max(log_magnitude, axis=[1], keepdims=True)
        # norm_mag = (log_magnitude - min_val) / tf.maximum(max_val - min_val, 1e-6)

        # # Devolver dos canales reales: [real, imag]
        # real_part = norm_mag * tf.cos(phase)
        # imag_part = norm_mag * tf.sin(phase)

        real_part = tf.math.real(fft_result)
        imag_part = tf.math.imag(fft_result)

        out = tf.stack([real_part, imag_part], axis=-1)
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

        x = input_layer
        x = keras.layers.Flatten()(x)

        # Ahora divido es sub capas de amplitud, frecuencia carrier e indice de modulacion
        f_c = keras.layers.Dense(self.output_shape // 3, activation="relu", name="f_C")(
            x
        )  # frecuencia carrier
        A = keras.layers.Dense(self.output_shape // 3, activation="tanh", name="A")(
            x
        )  # amplitud
        f_m = keras.layers.Dense(self.output_shape // 3, activation="relu", name="f_M")(
            x
        )

        print(f"f_C shape: {f_c.shape}, f_M shape: {f_m.shape}, A shape: {A.shape}")
        # ahora junto todo porque me pide una salida
        output_layer = FFTLayer(name="fft_layer")([f_c, A, f_m])
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    def compile(self, learning_rate=0.001):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss="mse")
        pass

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


def complex_mse(y_true, y_pred):
    real = tf.math.real(y_true) - tf.math.real(y_pred)
    imag = tf.math.imag(y_true) - tf.math.imag(y_pred)
    return tf.square(real) + tf.square(imag)


if __name__ == "__main__":
    sr = FS
    input_shape = (58797, 2)
    output_shape = 60

    model = FM_red(input_shape, output_shape)
    model.compile()

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
    mag = tf.math.abs(Y) + 1e-6
    phase = tf.math.angle(Y)

    log_mag = 20 * tf.math.log(tf.maximum(mag, 1e-6)) / tf.math.log(10.0)
    min_val = tf.reduce_min(log_mag)
    max_val = tf.reduce_max(log_mag)
    norm_mag = (log_mag - min_val) / tf.maximum(max_val - min_val, 1e-6)
    real_part = norm_mag * tf.cos(phase)
    imag_part = norm_mag * tf.sin(phase)
    y_stack = tf.stack([real_part, imag_part], axis=-1)

    # Igualar LARGO
    target_len = LARGO // 2 + 1
    y_stack = (
        y_stack[:target_len]
        if tf.shape(y_stack)[0] >= target_len
        else tf.pad(y_stack, [[0, target_len - tf.shape(y_stack)[0]], [0, 0]])
    )

    y_train = y_stack[tf.newaxis, ...].numpy().astype(np.float32)  # (1, target_len, 2)
    print(f"y_train shape: {y_train.shape}")

    history = model.fit(x_train, y_train, epochs=100, batch_size=1)

    model.save("modelo_fm.h5")

    # Plot history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    # Plot real and imaginary parts of the FFT
    # real
    y_pred = model.model.predict(x_train)[0]
    plt.plot(y_train[0, :, 0], label="Real Part")
    plt.plot(y_pred[:, 0], label="Predicted Real Part")
    plt.legend()
    # imaginary
    plt.subplot(2, 1, 2)
    plt.plot(y_train[0, :, 1], label="Imaginary Part")
    plt.plot(y_pred[:, 1], label="Predicted Imaginary Part")
    plt.legend()
    plt.grid()
    plt.show()
