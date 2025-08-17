import os
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt


def process_audio_file(path):
    # Takes fft
    y = librosa.load(path, sr=44100, mono=True)
    fft = tf.signal.fft(y)
    return fft


def process_folder(root_folder, save_path="dataset.npz"):
    X = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                full_path = os.path.join(root, file)
                try:
                    # Guardamos fft
                    print(f"Processing: {full_path}")
                    segments = process_audio_file(full_path)
                    X.extend(segments)
                except Exception as e:
                    print(f"Failed to process {full_path}: {e}")
    np.savez_compressed(save_path, X=tf.convert_to_tensor(X))
    print(f"Saved dataset to {save_path}. Total samples: {tf.shape(X)}")


if __name__ == "__main__":
    root_single = "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/red_simple/db"
    process_folder(root_single, save_path="dataset_single.npz")

