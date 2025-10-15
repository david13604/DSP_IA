import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import soundfile as sf
from red_magenta import Autoencder 

def plot_tiempo(real, pred):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title("Audio real ")
    plt.plot(real)
    plt.xlabel("muestras")
    plt.ylabel("Amplitud")

    plt.subplot(2, 1, 2)
    plt.title("Audio generado ")
    plt.plot(pred)
    plt.xlabel("muestras")
    plt.ylabel("Amplitud")

    plt.tight_layout()
    plt.show()

def plot_frec(real, pred):
    N = len(real)
    freqs = np.fft.rfftfreq(N, 1/16000)

    fft_real = np.abs(np.fft.rfft(real))
    fft_pred = np.abs(np.fft.rfft(pred))

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title("FFT del audio real")
    plt.plot(freqs, 20*np.log10(fft_real + 1e-8), color='royalblue')
    plt.ylabel("Magnitud [dB]")

    plt.subplot(2, 1, 2)
    plt.title("FFT del audio generado")
    plt.plot(freqs, 20*np.log10(fft_pred + 1e-8), color='orange')
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud [dB]")

    plt.tight_layout()
    plt.show()

model = Autoencder(
    input_shape=(None, 13),
    z_dim=16,
    n_harmonics=101,
    sample_rate=16000,
    n_samples=64000,
    stft_frame_length=1024,
    stft_frame_step=256
)
model.build()
model.autoencoder = keras.models.load_model("ddsp_autoencoder.h5", compile=False)


data = np.load('Chord_C_minor.npz')

audio_real = data["audio"].astype("float32")
mfcc = data["mfcc"].T.astype("float32")[np.newaxis, ...]      # (1, T, 13)
f0_norm = data["f0_norm"][:, np.newaxis].astype("float32")[np.newaxis, ...]  # normalizado
loud_norm = data["loudness_db_norm"][:, np.newaxis].astype("float32")[np.newaxis, ...]  # normalizado
f0_hz = data["f0_hz"][:, np.newaxis].astype("float32")[np.newaxis, ...]  # real Hz
loud_lin = data["loudness_lin"][:, np.newaxis].astype("float32")[np.newaxis, ...]  # real lineal


z = model.encoder(mfcc)  # (1, T, 16)
f0_norm = tf.constant(f0_norm, dtype=tf.float32)
loud_norm = tf.constant(loud_norm, dtype=tf.float32)
harmonics, noise = model.decoder([f0_norm, loud_norm, z])

print("harmonics:", harmonics.shape)
print("noise:", noise.shape)


audio_pred = model.additive_synth(f0_hz, loud_lin, harmonics, noise)
audio_pred = np.array(audio_pred[0])  

sf.write("guitar_to_DPA138.wav", audio_pred, 16000)
plot_tiempo(audio_real, audio_pred)
plot_frec(audio_real, audio_pred)