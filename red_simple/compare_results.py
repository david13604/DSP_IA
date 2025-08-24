import keras
import scipy
from scipy.signal import stft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
<<<<<<< HEAD
from entrenar_magnitud import FFTLayer
from espectro_pollo import savitzky_golay


def peak_envelope(mag, freqs):
    # Find peaks
    peaks, _ = scipy.signal.find_peaks(mag)
    # Interpolate envelope through peaks
    envelope_interp = scipy.interpolate.interp1d(
        freqs[peaks], mag[peaks], kind="cubic", fill_value="extrapolate"
    )
    return envelope_interp(freqs)

=======
from entrenar import FFTLayer
>>>>>>> 981f97f57522cb56e1361d4f10f28dd0eb0751bb

os.environ["KERAS_BACKEND"] = "tensorflow"

LARGO = 58796 * 2
FS = 44100

<<<<<<< HEAD
window_size = 51
order = 3
=======
model = keras.models.load_model("modelo_fm.h5", compile=False, custom_objects={"FFTLayer": FFTLayer})

def save_results(predictions, filename="predictions.npz"):
    np.savez_compressed(filename, predictions=predictions)
    print(f"Results saved to {filename}")
>>>>>>> 981f97f57522cb56e1361d4f10f28dd0eb0751bb

if __name__ == "__main__":
    sr = FS
    input_shape = (58797, 2)
    output_shape = 30

    # Load data for training
    path = "dataset_single.npz"

<<<<<<< HEAD
Y = tf.signal.rfft(tf.cast(y, tf.float32))
mag = tf.math.abs(Y)

smooth_mag = tf.abs(peak_envelope(mag, np.fft.rfftfreq(len(y), d=1.0/FS)))
y_train = tf.expand_dims(smooth_mag, axis=0)
=======
    x_train = np.load(path)["X"]
    # Reshape to match input shape
    x_train = np.stack([np.real(x_train), np.imag(x_train)], axis=-1).astype(np.float32)
    x_train = np.expand_dims(x_train, axis=0)

    print(f"x_train shape: {x_train.shape}")
>>>>>>> 981f97f57522cb56e1361d4f10f28dd0eb0751bb

    #path_y = ("/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/red_simple/Pollo_scream.mp3")
    path_y = (r"C:\Users\usuario\Desktop\DSP_IA_local\red_simple\db\guitar-single-note-d_120bpm_C_minor.wav")

    y, sr = librosa.load(path_y, sr=FS, mono=True)

<<<<<<< HEAD
# Predict once
y_pred = model.predict(y_train, verbose=0)[0]
freqs = np.fft.rfftfreq(LARGO, d=1.0 / FS)
y_train_np = tf.squeeze(y_train, axis=0).numpy()
scale = np.max(y_train_np)
plt.figure(figsize=(10, 4))
plt.plot(freqs, y_train_np / scale, label="|Y| (target)")
plt.plot(freqs, y_pred / scale, label="|Y_pred|")
plt.plot(freqs, tf.abs(peak_envelope(y_pred, freqs)) / scale, label="|Y_pred smooth|")
plt.xlim(0, 5000)
plt.ylim(0, 1.5)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
=======
    # Make target FFT use the same length as the model (LARGO)
    y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    n = tf.shape(y_tf)[0]
    start = tf.maximum((n - LARGO) // 2, 0)  # center crop if longer
    end = tf.minimum(start + LARGO, n)
    y_seg = y_tf[start:end]
    y_seg = tf.pad(y_seg, [[0, tf.maximum(LARGO - tf.shape(y_seg)[0], 0)]])  # zero-pad if shorter

    # RFFT on exactly LARGO samples so bins align
    Y = tf.signal.rfft(y_seg, fft_length=[LARGO])

    mag = tf.abs(Y)
    global_max_mag = tf.reduce_max(mag)
    global_max_mag = tf.maximum(global_max_mag, 1e-9)

    real_norm = tf.math.real(Y) / global_max_mag
    imag_norm = tf.math.imag(Y) / global_max_mag

    y_stack = tf.stack([real_norm, imag_norm], axis=-1)
    y_train = y_stack[tf.newaxis, ...].numpy().astype(np.float32)

    print(f"y_train shape: {y_train.shape}")

    # Plot
    plt.figure(figsize=(10, 6))

    # Predict once
    y_pred = model.predict(x_train)[0]


    save_results(y_pred, filename="predictions.npz")
    results = np.load("predictions.npz")["predictions"]
    print(f"Loaded predictions shape: {results.shape}")
    import pandas as pd
    df = pd.DataFrame(results, columns=[f"Bin_{i}" for i in range(results.shape[1])])
    print(df.head())
    # Frequency axis for rFFT bins
    freqs = np.fft.rfftfreq(LARGO, d=1.0/FS)

    # Magnitudes (numpy)
    true_real = y_train[0, :, 0]
    true_imag = y_train[0, :, 1]
    pred_real = y_pred[:, 0]
    pred_imag = y_pred[:, 1]

    true_mag = np.sqrt(true_real**2 + true_imag**2)
    pred_mag = np.sqrt(pred_real**2 + pred_imag**2)

    # Plot real
    plt.subplot(3, 1, 1)
    plt.plot(freqs, true_real, label="Real")
    plt.plot(freqs, pred_real, label="Pred Real", alpha=0.8)
    plt.xlim(0, 10000)  # focus band (adjust as needed)
    plt.grid(); plt.legend()

    # Plot imag
    plt.subplot(3, 1, 2)
    plt.plot(freqs, true_imag, label="Imag")
    plt.plot(freqs, pred_imag, label="Pred Imag", alpha=0.8)
    plt.xlim(0, 10000)
    plt.grid(); plt.legend()

    # Plot magnitude
    plt.subplot(3, 1, 3)
    plt.plot(freqs, true_mag, label="|Y|")
    plt.plot(freqs, pred_mag, label="|Y_pred|", alpha=0.8)
    plt.xlim(0, 10000)
    plt.grid(); plt.legend()
    plt.tight_layout()
    plt.show()
>>>>>>> 981f97f57522cb56e1361d4f10f28dd0eb0751bb
