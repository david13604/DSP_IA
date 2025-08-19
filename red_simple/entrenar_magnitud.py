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
        f_C, A, beta = inputs

        f_min = tf.constant(300.0, dtype=tf.float32)
        f_max = tf.constant(fs / 2.0, dtype=tf.float32)

        log_f_min = tf.math.log(f_min)
        log_f_max = tf.math.log(f_max)

        f_M = tf.constant(223, dtype=tf.float32)

        t = tf.range(largo, dtype=tf.float32) / tf.cast(fs, tf.float32)
        t = tf.reshape(t, (1, 1, -1))  # para broadcasting

        margin = tf.constant(0.95, dtype=tf.float32)  # keep away from hard edges
        #s_c = (tf.tanh(f_C) * margin + 1.0) * 0.5
        A = tf.nn.tanh(A)

        log_f_C = log_f_min + f_C * (log_f_max - log_f_min)
        f_C = tf.exp(log_f_C)

        sorted_indices = tf.argsort(log_f_C, axis=1)
        log_f_C = tf.gather(log_f_C, sorted_indices, batch_dims=1)
        A = tf.gather(A, sorted_indices, batch_dims=1)
        beta = tf.gather(beta, sorted_indices, batch_dims=1)

        f_C = tf.expand_dims(f_C, -1)
        beta = tf.expand_dims(beta, -1)
        A = tf.expand_dims(A, -1)

        beta = tf.nn.softplus(beta) * (2 * tf.constant(np.pi, dtype=tf.float32))

        mod = tf.sin(2 * tf.constant(np.pi, dtype=tf.float32) * f_M * t)
        fm_signal = tf.reduce_sum(
            A * tf.sin(2 * tf.constant(np.pi, dtype=tf.float32) * f_C * t + beta * mod),
            axis=1,
        )

        fft_result = tf.signal.rfft(tf.cast(fm_signal, tf.float32), fft_length=[largo])

        # Predict only normalized magnitude
        mag = tf.abs(fft_result)
        global_max_mag = tf.maximum(tf.reduce_max(mag), 1e-9)
        mag_norm = mag / global_max_mag

        mag_norm = tf.where(tf.math.is_finite(mag_norm), mag_norm, tf.zeros_like(mag_norm))
        return mag_norm  # [batch, bins] only magnitude


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

        # simple dense stack
        x = keras.layers.Dense(512, activation="relu")(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(128, activation="relu")(x)

        # Ahora divido en sub capas de amplitud, frecuencia carrier e indice de modulacion
        f_c = keras.layers.Dense(self.output_shape // 3, activation="sigmoid", name="f_C",bias_initializer=keras.initializers.Constant(1))(x)
        A = keras.layers.Dense(self.output_shape // 3, activation=None, name="A")(x)
        beta = keras.layers.Dense(self.output_shape // 3, activation=None, name="beta")(x)

        print(f"f_C shape: {f_c.shape}, beta shape: {beta.shape}, A shape: {A.shape}")

        # Output only magnitude spectrum
        output_layer = FFTLayer(name="fft_layer")([f_c, A, beta])
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    def compile(self, learning_rate=0.0003):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=2.0)
        self.model.compile(optimizer=optimizer, loss=symmetric_peak_loss)

    def fit(self, x_train, y_train, epochs, batch_size=1, validation_data=None):
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


def mag_loss(y_true, y_pred):
    # L1 on magnitude only
    return tf.reduce_mean(tf.square(y_true - y_pred))

def symmetric_peak_loss(y_true, y_pred, alpha=2.0):
    # emphasize both true and predicted peaks
    return tf.reduce_mean(tf.abs(tf.pow(y_true, alpha) - tf.pow(y_pred, alpha)))


if __name__ == "__main__":
    sr = FS
    output_shape = 90

    # Load data for training
    path = "dataset_single.npz"

    x_train = np.load(path)["X"]
    # Use one channel: magnitude
    x_train = np.abs(x_train).astype(np.float32)
    x_train = x_train[..., np.newaxis]        # (time, 1)
    x_train = np.expand_dims(x_train, axis=0) # (1, time, 1)

    path_y = (
        "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/red_simple/Pollo_scream.mp3"
    )

    input_shape = x_train.shape[1:]
    print(f"x_train shape: {x_train.shape}")

    y, sr = librosa.load(path_y, sr=FS, mono=True)

    # Make target FFT use the same length as the model (LARGO)
    y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    n = tf.shape(y_tf)[0]
    start = tf.maximum((n - LARGO) // 2, 0)  # center crop if longer
    end = tf.minimum(start + LARGO, n)
    y_seg = y_tf[start:end]
    y_seg = tf.pad(y_seg, [[0, tf.maximum(LARGO - tf.shape(y_seg)[0], 0)]])  # zero-pad if shorter

    # RFFT on exactly LARGO samples so bins align
    Y = tf.signal.rfft(y_seg, fft_length=[LARGO])

    # Target: normalized magnitude only
    Y = tf.abs(Y)
    global_max_mag = tf.maximum(tf.reduce_max(Y), 1e-9)

    y_train = Y / global_max_mag
    y_train = tf.expand_dims(y_train, axis=0)

    print(f"y_train shape (magnitude only): {y_train.shape}")

    model = FM_red(input_shape, output_shape)
    model.compile()

    history = model.fit(x_train, y_train, epochs=20, batch_size=1)

    model.save("modelo_fm.h5")

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Predict once
    y_pred = model.model.predict(x_train, verbose=0)[0]  # [bins]

    # Frequency axis for rFFT bins
    freqs = np.fft.rfftfreq(LARGO, d=1.0/FS)

    # Plot magnitude (target vs prediction)
    y_train_np = tf.squeeze(y_train, axis=0).numpy()
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, y_train_np, label="|Y| (target)")
    plt.plot(freqs, y_pred, label="|Y_pred|", alpha=0.8)
    plt.xlim(0, 10000)
    plt.grid(); plt.legend()
    plt.tight_layout()
    plt.show()
   