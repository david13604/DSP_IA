import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import librosa
from entrenar_magnitud import FFTLayer, peak_envelope_tf


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

if __name__ == "__main__":
    sr = FS
    output_shape = 30

    # Load data for training
    path_y = "/mnt/c/Users/matth/Desktop/Other/DSP_IA/red_simple/Pollo_scream.mp3"

    y, sr = librosa.load(path_y, sr=FS, mono=True)

    Y = tf.signal.rfft(tf.cast(y, tf.float32))
    mag = tf.math.abs(Y)

    smooth_mag = tf.abs(peak_envelope_tf(mag))
    y_train = tf.expand_dims(smooth_mag, axis=0)
    input_shape = y_train.shape[1:]

    # load model
    model = keras.models.load_model(
        "/mnt/c/Users/matth/Desktop/Other/DSP_IA/red_simple/modelo_fm.h5",
        custom_objects={"FFTLayer": FFTLayer},
        compile=False,
    )

    # Predict once
    y_pred = model.predict(y_train, verbose=0)[0]
    freqs = np.fft.rfftfreq(LARGO, d=1.0 / FS)
    y_train_np = tf.squeeze(y_train, axis=0).numpy()
    scale = np.max(y_train_np)
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, y_train_np / scale, label="|Y| (target)")
    plt.plot(freqs, y_pred / scale, label="|Y_pred|")
    plt.plot(freqs, tf.abs(peak_envelope_tf(y_pred)) / scale, label="|Y_pred smooth|")
    plt.xlim(0, 5000)
    plt.ylim(0, 1.5)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
