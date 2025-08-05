import keras
import os
from scipy.signal import stft
import numpy as np
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"


class STFTLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(STFTLayer, self).__init__(**kwargs)

    def call(self, inputs, fs=44100, cant_muestras=300):
        cant_muestras = int(cant_muestras * 44100 // 1000)
        f_C, I, A = inputs

        print(f"f_C shape: {f_C.shape}, I shape: {I.shape}, A shape: {A.shape}")
        t = tf.linspace(0.0, (cant_muestras - 1) / fs, cant_muestras)
        t = tf.reshape(t, (1, 1, -1))  # para broadcasting
        f_C = tf.expand_dims(f_C, -1)
        I = tf.expand_dims(I, -1)
        A = tf.expand_dims(A, -1)

        mod = tf.sin(2 * np.pi * f_C * t)
        fm_signal = tf.reduce_sum(A * tf.sin(2 * np.pi * f_C * t + I * mod), axis=1)

        stft_result = tf.signal.stft(
            fm_signal, frame_length=256, frame_step=128, fft_length=256
        )
        magnitude = tf.abs(stft_result)
        log_magnitude = (
            20 * tf.math.log(tf.maximum(magnitude, 1e-6)) / tf.math.log(10.0)
        )

        min_val = tf.reduce_min(log_magnitude, axis=[1, 2], keepdims=True)
        max_val = tf.reduce_max(log_magnitude, axis=[1, 2], keepdims=True)
        norm_mag = (log_magnitude - min_val) / tf.maximum(max_val - min_val, 1e-6)

        return norm_mag


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

        # Ahora la CNN tipica
        x = keras.layers.Conv2D(32, (3, 3), activation="relu")(input_layer)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(self.output_shape, activation="relu")(x)

        # Ahora divido es sub capas de amplitud, frecuencia carrier e indice de modulacion
        f_c = keras.layers.Dense(
            self.output_shape // 3, activation="sigmoid", name="f_C"
        )(
            x
        )  # frecuencia carrier
        I = keras.layers.Dense(
            self.output_shape // 3, activation="sigmoid", name="f_M"
        )(
            x
        )  # indice de modulacion
        A = keras.layers.Dense(
            self.output_shape // 3, activation="sigmoid", name="f_I"
        )(
            x
        )  # amplitud

        print(f"f_C shape: {f_c.shape}, I shape: {I.shape}, A shape: {A.shape}")
        # ahora junto todo porque me pide unsa salida
        output_layer = STFTLayer(name="stft_layer")([f_c, I, A])
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    def compile(self, learning_rate=0.0001):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss="mse")
        pass

    def fit(self, x_train, epochs=10, batch_size=32, validation_data=None):
        self.model.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            shuffle=True,
        )
        pass


if __name__ == "__main__":
    sr = 44100
    length = 300
    input_shape = (102, 129, 1)
    output_shape = 60

    model = FM_red(input_shape, output_shape)
    model.compile()

    # Load data for training
    path = "dataset.npz"

    x_train = np.load(path)["X"]
    # Reshape to match input shape
    x_train = np.expand_dims(x_train, axis=-1).astype(
        np.float32
    )  # Add channel dimension

    print(f"x_train shape: {x_train.shape}")
    print(np.isfinite(x_train).all())  # Check for NaN or Inf values

    model.fit(x_train, epochs=5, batch_size=64)
