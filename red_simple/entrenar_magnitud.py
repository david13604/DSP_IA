import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import librosa
from espectro_pollo import savitzky_golay

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"Using GPU(s): {[g.name for g in gpus]}")
    except Exception as e:
        print("Could not set memory growth:", e)
else:
    print("No GPU detected, running on CPU.")

LARGO = 12407 * 2
FS = 44100

window_size = 51
order = 3


def peak_envelope_1d(mag_1d: tf.Tensor):
    mag_1d = tf.convert_to_tensor(mag_1d, dtype=tf.float32)

    # Neighbor compare
    left = tf.concat([mag_1d[:1], mag_1d[:-1]], axis=0)
    right = tf.concat([mag_1d[1:], mag_1d[-1:]], axis=0)
    peak_mask = (mag_1d >= left) & (mag_1d >= right)

    peaks = tf.where(peak_mask)[:, 0]  # indices of local maxima
    # Ensure endpoints
    n = tf.shape(mag_1d)[0]
    endpoints = tf.convert_to_tensor([0, n - 1], dtype=tf.int64)
    peaks = tf.concat([peaks, endpoints], axis=0)
    peaks = tf.sort(tf.unique(peaks).y)

    # Need at least two distinct peaks for interpolation
    num_peaks = tf.shape(peaks)[0]

    def single_peak():
        return mag_1d  # Cannot interpolate, return original

    def interpolate():
        peak_vals = tf.gather(mag_1d, peaks)
        x_full = tf.range(n, dtype=tf.int32)

        # For each x, find right peak index
        right_idx = tf.searchsorted(tf.cast(peaks, tf.int32), x_full, side="right")
        right_idx = tf.clip_by_value(right_idx, 1, tf.shape(peaks)[0] - 1)
        left_idx = right_idx - 1

        x0 = tf.cast(tf.gather(peaks, left_idx), tf.float32)
        x1 = tf.cast(tf.gather(peaks, right_idx), tf.float32)
        y0 = tf.gather(peak_vals, left_idx)
        y1 = tf.gather(peak_vals, right_idx)

        denom = tf.where(x1 > x0, x1 - x0, tf.ones_like(x1))
        t = tf.where(x1 > x0, (tf.cast(x_full, tf.float32) - x0) / denom, 0.0)
        return y0 + t * (y1 - y0)

    return tf.cond(num_peaks < 2, single_peak, interpolate)


def peak_envelope_tf(mag):
    mag = tf.convert_to_tensor(mag, dtype=tf.float32)
    rank = tf.rank(mag)

    def rank1():
        return peak_envelope_1d(mag)

    def higher():
        n = tf.shape(mag)[-1]
        flat = tf.reshape(mag, (-1, n))  # (M, N)
        env = tf.map_fn(
            lambda row: peak_envelope_1d(row), flat, fn_output_signature=tf.float32
        )
        return tf.reshape(env, tf.shape(mag))

    return tf.cond(tf.equal(rank, 1), rank1, higher)


def topfig(x=10, y=10):
    figmgr = plt.get_current_fig_manager()

    # Try to get current size (fallback if not available)
    try:
        w, h = figmgr.canvas.get_width_height()
    except Exception:
        w, h = 800, 600

    # Tkinter (TkAgg)
    if hasattr(figmgr, "window") and hasattr(figmgr.window, "geometry"):
        try:
            figmgr.window.geometry(f"{w}x{h}+{x}+{y}")
            # Replacement for former: figmgr.canvas.manager.window.raise_()
            if hasattr(figmgr.window, "lift"):
                figmgr.window.lift()
            # Briefly set topmost to bring to front (then revert)
            if hasattr(figmgr.window, "attributes"):
                figmgr.window.attributes("-topmost", True)
                figmgr.window.after(
                    50, lambda: figmgr.window.attributes("-topmost", False)
                )
        except Exception:
            pass

    # Qt (Qt5Agg / QtAgg)
    elif hasattr(figmgr, "window") and hasattr(figmgr.window, "move"):
        try:
            figmgr.window.move(x, y)
            if hasattr(figmgr.window, "raise_"):
                figmgr.window.raise_()
            if hasattr(figmgr.window, "activateWindow"):
                figmgr.window.activateWindow()
        except Exception:
            pass

    # WX
    elif hasattr(figmgr, "frame") and hasattr(figmgr.frame, "SetPosition"):
        try:
            figmgr.frame.SetPosition((x, y))
            if hasattr(figmgr.frame, "Raise"):
                figmgr.frame.Raise()
        except Exception:
            pass


class FFTLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FFTLayer, self).__init__(**kwargs)

    def call(self, inputs, fs=FS, largo=LARGO):
        f_C, A, beta = inputs

        f_min = tf.constant(100.0, dtype=tf.float32)
        f_max = tf.constant(fs / 2.0, dtype=tf.float32)

        log_f_min = tf.math.log(f_min)
        log_f_max = tf.math.log(f_max)

        f_M = tf.constant(101, dtype=tf.float32)

        t = tf.range(largo, dtype=tf.float32) / tf.cast(fs, tf.float32)
        t = tf.reshape(t, (1, 1, -1))  # para broadcasting

        # Limit f_C
        # f_C = tf.clip_by_value(f_C, log_f_min, log_f_max)

        n_carriers = tf.cast(tf.shape(A)[1], tf.float32)
        A = tf.nn.tanh(A) / tf.sqrt(n_carriers)

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
        mag = tf.abs(fft_result)
        mag = tf.where(tf.math.is_finite(mag), mag, tf.zeros_like(mag))
        return mag


class FM_red:
    def __init__(self, input_shape, output_shape, max) -> None:
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
            kernel_initializer=keras.initializers.Zeros(),
            bias_initializer=keras.initializers.Constant(750.0),
        )(x)
        A = keras.layers.Dense(
            self.output_shape // 3,
            activation=None,
            name="A",
            kernel_initializer=keras.initializers.Zeros(),
            bias_initializer=keras.initializers.Constant(1.0),
        )(x)
        beta = keras.layers.Dense(
            self.output_shape // 3,
            activation=None,
            name="beta",
            kernel_initializer=keras.initializers.Zeros(),
            bias_initializer=keras.initializers.Constant(0.0),
        )(x)

        print(f"f_C shape: {f_c.shape}, beta shape: {beta.shape}, A shape: {A.shape}")

        # Output only magnitude spectrum
        output_layer = FFTLayer(name="fft_layer")([f_c, A, beta])
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    def compile(self, learning_rate=0.0001):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=2.0)
        self.model.compile(optimizer=optimizer, loss=mag_loss)

    def fit(self, x_train, y_train, epochs, batch_size=1, validation_data=None):
        y_train_np = tf.squeeze(y_train, axis=0).numpy()

        freqs = np.fft.rfftfreq(LARGO, d=1.0 / FS)
        scale = np.max(y_train_np)
        plot_callback = SpectrumPlotCallback(y_train, freqs, scale)
        callbacks = [
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=25, min_lr=1e-6, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="loss", patience=250, restore_best_weights=False, verbose=1
            ),
            plot_callback,
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


class SpectrumPlotCallback(keras.callbacks.Callback):
    def __init__(self, x_sample, freqs, scale):
        super().__init__()
        self.x_sample = x_sample
        self.freqs = freqs
        self.scale = scale
        self.pause = 0.5

    def on_epoch_end(self, epoch, logs=None):
        # Predict
        y_pred = self.model.predict(self.x_sample, verbose=0)[0]  # (N,)
        # True target (same as input in this setup)
        y_true = tf.squeeze(self.x_sample, axis=0).numpy()  # (N,)
        # Envelope of prediction
        y_pred_env = tf.abs(peak_envelope_tf(y_pred))
        # Ensure shapes match freq axis
        if y_true.ndim > 1:
            y_true = np.squeeze(y_true)
        if y_pred_env.ndim > 1:
            y_pred_env = np.squeeze(y_pred_env)
        plt.figure(figsize=(10, 4))
        # plt.plot(self.freqs, y_pred/self.scale, label=f"|Y_pred| epoch {epoch+1}")
        plt.plot(self.freqs, y_true / self.scale, label="|Y_train|")
        plt.plot(self.freqs, y_pred_env / self.scale, label="|Y_pred envelope|")
        plt.xlim(0, 10000)
        plt.grid()
        plt.legend()
        plt.title(f"Spectrum after epoch {epoch+1}")
        plt.tight_layout()
        topfig()
        plt.show(block=False)
        plt.pause(self.pause)
        plt.close()


def mag_loss(y_true, y_pred):
    # L1
    y_pred = tf.abs(peak_envelope_tf(y_pred))
    return tf.reduce_mean(tf.square(y_true - y_pred))


if __name__ == "__main__":
    sr = FS
    output_shape = 3

    path_y = "/mnt/c/Users/matth/Desktop/Other/DSP_IA/red_simple/Pollo_scream.mp3"

    y, sr = librosa.load(path_y, sr=44100, mono=True)

    Y = tf.signal.rfft(tf.cast(y, tf.float32))
    mag = tf.math.abs(Y)

    smooth_mag = tf.abs(savitzky_golay(mag, window_size, order))
    y_train = tf.expand_dims(smooth_mag, axis=0)

    max = tf.reduce_max(mag)

    input_shape = y_train.shape[1:]

    print(f"y_train shape (magnitude only): {y_train.shape}")

    model = FM_red(input_shape, output_shape, max)
    model.compile()

    history = model.fit(y_train, y_train, epochs=2000, batch_size=1)

    model.save("modelo_fm.h5")

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Predict once
    y_pred = model.model.predict(y_train, verbose=0)[0]
    y_pred = np.array(y_pred)
    freqs = np.fft.rfftfreq(LARGO, d=1.0 / FS)
    y_train_np = tf.squeeze(y_train, axis=0).numpy()
    scale = np.max(y_train_np)
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, y_train_np / scale, label="|Y| (target)")
    plt.plot(freqs, y_pred / scale, label="|Y_pred|")
    plt.plot(freqs, tf.abs(peak_envelope_tf(y_pred)) / scale, label="|Y_pred smooth|")
    plt.xlim(0, 10000)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
