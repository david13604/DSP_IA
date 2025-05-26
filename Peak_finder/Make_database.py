import os
import numpy as np
import librosa
import scipy.signal
from scipy.interpolate import interp1d


def extract_peaks_from_spectrum(mag, freqs, n_peaks=15):
    norm_mag = mag / np.max(mag + 1e-12)
    db_mag = 20 * np.log10(norm_mag + 1e-12)
    peaks, _ = scipy.signal.find_peaks(db_mag, distance=75, prominence=6)
    peak_tuples = [(norm_mag[p], freqs[p]) for p in peaks]
    peak_tuples.sort(reverse=True)
    top_peaks = peak_tuples[:n_peaks]
    if len(top_peaks) < n_peaks:
        pad = [(0.0, 0.0)] * (n_peaks - len(top_peaks))
        top_peaks += pad
    peak_amps, peak_freqs = zip(*top_peaks)
    return np.array(peak_freqs), np.array(peak_amps)


def compute_spectrum(audio_path, rate=44100, target_duration=4):
    y, sr = librosa.load(audio_path, sr=rate, mono=True)
    target_len = int(rate * target_duration)
    if len(y) < target_len:
        # Zero padding
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    elif len(y) > target_len:
        # Acortar
        y = y[:target_len]
    n = len(y)
    pad_len = 2 ** int(np.ceil(np.log2(n)))
    y_padded = np.zeros(pad_len)
    y_padded[:n] = y
    fft_vals = np.fft.fft(y_padded)
    freqs = np.linspace(0, sr, len(fft_vals) // 2)
    mag = np.abs(fft_vals[: len(fft_vals) // 2])
    return freqs, mag


def shift_spectrum(mag, shift_bins):
    shifted = np.roll(mag, shift_bins)
    if shift_bins > 0:
        shifted[:shift_bins] = 0
    elif shift_bins < 0:
        shifted[shift_bins:] = 0
    return shifted


def stretch_spectrum(mag, stretch_factor):
    n = len(mag)
    x_original = np.linspace(0, 1, n)
    # Enscanchar o comprimir con interpolación lineal
    x_stretched = np.linspace(0, 1, int(n / stretch_factor))
    interp = interp1d(x_original, mag, kind="linear", bounds_error=False, fill_value=0)
    mag_stretched = interp(x_stretched)
    # Volver al largo original
    x_resample = np.linspace(0, 1, n)
    interp_resample = interp1d(
        np.linspace(0, 1, len(mag_stretched)),
        mag_stretched,
        kind="linear",
        bounds_error=False,
        fill_value=0,
    )
    stretched = interp_resample(x_resample)
    return stretched


def process_audio_file(path, rate=44100, n_peaks=15, augment=True):
    freqs, mag = compute_spectrum(path, rate=rate)
    inputs = []
    outputs = []

    # Original
    peak_freqs, peak_amps = extract_peaks_from_spectrum(mag, freqs, n_peaks)
    inputs.append(mag / np.max(mag + 1e-12))
    outputs.append(
        np.stack([peak_freqs / freqs[-1], peak_amps], axis=1)
    )  # Frecuencias normalizadas

    if augment:
        shifts = [-2500, -2000, -1500, -1000, -500, 500, 1000, 1500, 2000, 2500]
        stretches = [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]
        for s in shifts:
            mag_shifted = shift_spectrum(mag, s)
            pf, pa = extract_peaks_from_spectrum(mag_shifted, freqs, n_peaks)
            inputs.append(mag_shifted / np.max(mag_shifted + 1e-12))
            outputs.append(np.stack([pf / freqs[-1], pa], axis=1))

        for sf in stretches:
            mag_stretched = stretch_spectrum(mag, sf)
            pf, pa = extract_peaks_from_spectrum(mag_stretched, freqs, n_peaks)
            inputs.append(mag_stretched / np.max(mag_stretched + 1e-12))
            outputs.append(np.stack([pf / freqs[-1], pa], axis=1))

    return inputs, outputs


def process_folder(
    root_folder, rate=44100, n_peaks=15, augment=True, save_path="dataset.npz"
):
    X = []
    Y = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                full_path = os.path.join(root, file)
                try:
                    print(f"Processing: {full_path}")
                    inps, outs = process_audio_file(full_path, rate, n_peaks, augment)
                    X.extend(inps)
                    Y.extend(outs)
                    #Print dimensions of inputs and outputs
                    print(f"Input shape: {[inp.shape for inp in inps]}")
                    print(f"Output shape: {[out.shape for out in outs]}")
                except Exception as e:
                    print(f"Failed to process {full_path}: {e}")
    np.savez_compressed(save_path, X=np.array(X), Y=np.array(Y))
    print(f"Saved dataset to {save_path}. Total samples: {len(X)}")

if __name__ == "__main__":
    root_folder = "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/SoundEffects"
    process_folder(root_folder, rate=44100, n_peaks=15, augment=True, save_path="dataset.npz")
