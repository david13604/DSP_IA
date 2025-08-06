import os
import tensorflow as tf
import numpy as np
import librosa
import scipy.signal
from scipy.interpolate import interp1d


def process_audio_file(path, length=300, jump=150):
    # Receives path to audio file and selects segments from it
    # length is time in ms
    # jump is the step in ms to select next segment
    length = int(length * 44100 / 1000)  # Convert to samples
    jump = int(jump * 44100 / 1000)  # Convert to samples
    y, sr = librosa.load(path, sr=44100, mono=True)
    segments = []
    for start in range(0, len(y) - length, jump):
        segment = y[start : start + length]
        if len(segment) < length:
            segment = tf.pad(segment, (0, length - len(segment)), mode="constant")
        # Take STFT
        stft_segment = tf.signal.stft(segment, frame_length=128, frame_step=32)
        segments.append(stft_segment)
    return tf.convert_to_tensor(segments)


def process_folder(root_folder, rate=44100, save_path="dataset.npz"):
    X = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                full_path = os.path.join(root, file)
                try:
                    print(f"Processing: {full_path}")
                    segments = process_audio_file(full_path, length=300, jump=150)
                    segments = tf.abs(segments) 
                    # Normalize
                    log_magnitude = (
                        20 * tf.math.log(tf.maximum(segments, 1e-6)) / tf.math.log(10.0)
                    )
                    min_val = tf.reduce_min(log_magnitude, axis=[1, 2], keepdims=True)
                    max_val = tf.reduce_max(log_magnitude, axis=[1, 2], keepdims=True)
                    norm_mag = (log_magnitude - min_val) / tf.maximum(max_val - min_val, 1e-6)
                    X.extend(norm_mag)
                    # Dimensiones
                    # print(f"Input shape: {[seg.shape for seg in segments]}")
                except Exception as e:
                    print(f"Failed to process {full_path}: {e}")
    print(X[0].shape)
    np.savez_compressed(save_path, X=tf.convert_to_tensor(X))
    print(f"Saved dataset to {save_path}. Total samples: {len(X)}")


if __name__ == "__main__":
    # root_folder = r"C:\Users\usuario\Desktop\DSP_IA_local\DSP_IA\SoundEffects"
    root_folder = "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/SoundEffects"
    #root_single = "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/SoundEffects/DavidDumais - ATV Arctic Cat 650 H1"
    root_single = "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/Particular"
    process_folder(root_folder, rate=44100, save_path="dataset.npz")
    process_folder(root_single, rate=44100, save_path="dataset_single.npz")
