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
        # Unpack
        f_C, A, beta, phi = inputs

        f_min = tf.constant(100.0, dtype=tf.float32)
        f_max = tf.constant(fs / 2.0, dtype=tf.float32)

        log_f_min = tf.math.log(f_min)
        log_f_max = tf.math.log(f_max)

        f_M = tf.constant(223, dtype=tf.float32)

        t = tf.range(largo, dtype=tf.float32) / tf.cast(fs, tf.float32)
        t = tf.reshape(t, (1, 1, -1))  # para broadcasting

        margin = tf.constant(0.95, dtype=tf.float32)  # keep away from hard edges
        #s_c = (tf.tanh(f_C) * margin + 1.0) * 0.5
        n_carriers = tf.cast(tf.shape(A)[1], tf.float32)
        A = tf.nn.tanh(A) / tf.sqrt(n_carriers + 1e-8)

        s_c = (tf.tanh(f_C) * margin + 1.0) * 0.5  # in (0.025, 0.975)
        log_f_C = log_f_min + s_c * (log_f_max - log_f_min)
        f_C = tf.exp(log_f_C)

        sorted_indices = tf.argsort(f_C, axis=1)
        f_C = tf.gather(f_C, sorted_indices, batch_dims=1)
        A = tf.gather(A, sorted_indices, batch_dims=1)
        beta = tf.gather(beta, sorted_indices, batch_dims=1)
        phi = tf.gather(phi, sorted_indices, batch_dims=1)

        f_C = tf.expand_dims(f_C, -1)
        beta = tf.expand_dims(beta, -1)
        A = tf.expand_dims(A, -1)
        phi = tf.expand_dims(phi, -1)

        # Constrain FM index
        beta = tf.nn.softplus(beta)
        beta = tf.clip_by_value(beta, 0.0, 8.0)

        # Map phase to [0, 2π]
        two_pi = tf.constant(2.0 * np.pi, dtype=tf.float32)
        phi = two_pi * tf.sigmoid(phi)

        mod = tf.sin(two_pi * f_M * t)
        fm_signal = tf.reduce_sum(
            A * tf.sin(two_pi * f_C * t + beta * mod + phi),
            axis=1,
        )

        # Hann window before FFT to reduce leakage/spikes
        w = tf.signal.hann_window(largo, dtype=tf.float32)
        w = tf.reshape(w, (1, -1))
        fm_signal = fm_signal * w

        fft_result = tf.signal.rfft(tf.cast(fm_signal, tf.float32), fft_length=[largo])

        # Predict raw magnitude
        mag = tf.abs(fft_result) + 1e-9
        mag = tf.where(tf.math.is_finite(mag), mag, tf.zeros_like(mag))
        return mag  # [batch, bins]

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
        # New: per-carrier initial phase
        phi = keras.layers.Dense(
            self.output_shape // 3,
            activation=None,
            name="phi",
            bias_initializer="zeros",
        )(x)

        print(f"f_C shape: {f_c.shape}, beta shape: {beta.shape}, A shape: {A.shape}")

        # Output only magnitude spectrum
        output_layer = FFTLayer(name="fft_layer")([f_c, A, beta, phi])
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

def symmetric_peak_loss(y_true, y_pred, alpha=3.0): 
    # emphasize both true and predicted peaks
    return tf.reduce_mean(tf.abs(tf.pow(y_true, alpha) - tf.pow(y_pred, alpha)))

def mag_loss(y_true, y_pred):
    # L1 on magnitude only
    return tf.reduce_mean(tf.square(y_true - y_pred))

def spectral_convergence(y_true, y_pred, eps=1e-9):
    # normalize by target max (fixed wrt prediction)
    scale = tf.reduce_max(y_true, axis=-1, keepdims=True) + eps
    y_true_n = y_true / scale
    y_pred_n = y_pred / scale
    num = tf.norm(y_true_n - y_pred_n, ord='euclidean', axis=-1)
    den = tf.norm(y_true_n, ord='euclidean', axis=-1) + eps
    return tf.reduce_mean(num / den)

def peak_distribution_kl(y_true, y_pred, gamma=8.0, eps=1e-9):
    # Soft peak matching: distributions emphasize large bins (peaks)
    scale = tf.reduce_max(y_true, axis=-1, keepdims=True) + eps
    y_true_n = y_true / scale
    y_pred_n = y_pred / scale
    p_true = tf.nn.softmax(gamma * y_true_n, axis=-1)
    p_pred = tf.nn.softmax(gamma * y_pred_n, axis=-1)
    kl = tf.reduce_sum(p_true * (tf.math.log(p_true + eps) - tf.math.log(p_pred + eps)), axis=-1)
    return tf.reduce_mean(kl)

def energy_match(y_true, y_pred, eps=1e-9):
    s_true = tf.reduce_sum(y_true, axis=-1)
    s_pred = tf.reduce_sum(y_pred, axis=-1)
    return tf.reduce_mean(tf.abs(s_pred - s_true) / (s_true + eps))

def log_mag_l1(y_true, y_pred, eps=1e-9):
    scale = tf.reduce_max(y_true, axis=-1, keepdims=True) + eps
    y_true_n = y_true / scale
    y_pred_n = y_pred / scale
    return tf.reduce_mean(tf.abs(tf.math.log(y_true_n + eps) - tf.math.log(y_pred_n + eps)))

def gaussian_blur1d(x, sigma=2.0):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    radius = tf.cast(tf.math.ceil(3.0 * sigma), tf.int32)
    size = 2 * radius + 1
    grid = tf.range(-radius, radius + 1, dtype=tf.float32)
    kernel = tf.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.reshape(kernel, (size, 1, 1))  # [W, 1, 1]
    x3 = tf.expand_dims(x, axis=-1)            # [B, bins, 1]
    x_blur = tf.nn.conv1d(x3, kernel, stride=1, padding="SAME")
    return tf.squeeze(x_blur, axis=-1)

def smoothed_log_mag_l1(y_true, y_pred, eps=1e-9, sigma=2.0):
    scale = tf.reduce_max(y_true, axis=-1, keepdims=True) + eps
    y_true_n = y_true / scale
    y_pred_n = y_pred / scale
    y_true_s = gaussian_blur1d(y_true_n, sigma=sigma)
    y_pred_s = gaussian_blur1d(y_pred_n, sigma=sigma)
    return tf.reduce_mean(tf.abs(tf.math.log(y_true_s + eps) - tf.math.log(y_pred_s + eps)))

def composite_spectral_loss(y_true, y_pred):
    return (
        0.25 * log_mag_l1(y_true, y_pred)
      + 0.25 * spectral_convergence(y_true, y_pred)
      + 0.30 * smoothed_log_mag_l1(y_true, y_pred, sigma=2.0)
      + 0.10 * peak_distribution_kl(y_true, y_pred)
      + 0.10 * energy_match(y_true, y_pred)
    )


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

    # Target: raw magnitude (loss will normalize consistently)
    y_train = tf.abs(Y) + 1e-9
    y_train = tf.expand_dims(y_train, axis=0)

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
    # ...existing code...
    y_pred = model.model.predict(x_train, verbose=0)[0]  # [bins]
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
# ...existing code...
   