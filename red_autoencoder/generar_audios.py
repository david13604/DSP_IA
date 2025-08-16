import numpy as np
import keras
import librosa
import sounddevice as sd

# Load autoencoder
model = keras.models.load_model("primero_autoencoder.h5", compile=False)

# Load STFTs
data = np.load("dataset_single.npz")
X = data["X"]  # shape: (n_samples, n_freq, n_time)
X = np.expand_dims(X, axis=-1).astype(np.float32)

# Select sample
n = 1
input_stft = X[n : n + 1]
reconstructed_stft = model.predict(input_stft)

# Remove channel dimension
input_stft = np.squeeze(input_stft)
reconstructed_stft = np.squeeze(reconstructed_stft)

# Invert STFT to audio
def stft_to_audio(stft_matrix, hop_length=512):
    # If magnitude only, use Griffin-Lim to estimate phase
    audio = librosa.griffinlim(stft_matrix, hop_length=hop_length)
    return audio

# Play original audio
audio_orig = stft_to_audio(input_stft)
print("Playing original audio...")
sd.play(audio_orig, samplerate=22050)
sd.wait()

# Play reconstructed audio
audio_recon = stft_to_audio(reconstructed_stft)
print("Playing reconstructed audio...")
sd.play(audio_recon, samplerate=22050)
sd.wait()