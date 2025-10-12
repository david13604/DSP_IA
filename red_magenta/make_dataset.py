import os
import numpy as np
import librosa

PATH = r'C:\Users\usuario\Desktop\DSP_IA_local\TU-Note_Violin\WAV'
OUTPUT_PATH = r'C:\Users\usuario\Desktop\DSP_IA_local\Dataset_Processed'
os.makedirs(OUTPUT_PATH, exist_ok=True)


def resample(filepath):
    """Carga un audio y lo reescala a 16 kHz."""
    data, sr = librosa.load(filepath, sr=96000)
    sr_new = 16000
    audio_resampled = librosa.resample(data, orig_sr=sr, target_sr=sr_new)
    return audio_resampled, sr_new


def slice_audio(x, sr=16000, segment_duration=4.0):
    """Divide un audio en segmentos de 4 segundos, rellenando si es más corto."""
    seg_len = int(sr * segment_duration)
    n_segments = int(np.ceil(len(x) / seg_len))
    slices = []

    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len
        segment = x[start:end]
        if len(segment) < seg_len:
            segment = np.pad(segment, (0, seg_len - len(segment)))
        slices.append(segment)

    return slices


def get_f0(x, sr, frame_length=2048, hop_length=256, fmin=50, fmax=2000):
    """Estimación de f0 por método cepstral (por frames)."""
    n_frames = len(x) // hop_length + 1
    f0s = []

    for i in range(n_frames):
        start = i * hop_length
        frame = x[start:start + frame_length]
        if len(frame) < frame_length:
            break
        frame = frame * np.hamming(len(frame))
        spectrum = np.fft.rfft(frame)
        log_mag = np.log(np.abs(spectrum) + 1e-10)
        cepstrum = np.fft.irfft(log_mag)
        qmin = int(sr / fmax)
        qmax = int(sr / fmin)
        peak_index = np.argmax(cepstrum[qmin:qmax]) + qmin
        f0 = sr / peak_index
        f0s.append(f0)

    f0_n = np.array(f0s)
    f0_midi = 69 + 12 * np.log2(f0_n / 440.0 + 1e-10)
    f0_norm = (f0_midi - np.mean(f0_midi)) / np.std(f0_midi)
    return f0_n, f0_norm


def loudness(data, sr=16000, n_fft=2048, hop_length=256):
    """Calcula loudness perceptual (A-weighted) normalizado y lineal."""
    stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=1024)
    stft_abs = np.abs(stft)
    power = stft_abs**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def a_weighting(f):
        f = np.array(f)
        ra = (12200**2 * f**4) / (
            (f**2 + 20.6**2)
            * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))
            * (f**2 + 12200**2)
        )
        A = 20 * np.log10(ra + 1e-12) + 2.0
        return A

    A_dB = a_weighting(freqs)
    A_lin = 10 ** (A_dB / 20)
    power_A = power * A_lin[:, np.newaxis]

    loudness_db = 10 * np.log10(np.mean(power_A, axis=0) + 1e-12)
    loudness_db_norm = (loudness_db - np.mean(loudness_db)) / np.std(loudness_db)
    loudness_lin = np.mean(power_A, axis=0)
    return loudness_db_norm, loudness_lin, stft_abs


def compute_mfcc(data, sr=16000, n_fft=2048, hop_length=256, n_mfcc=13):
    """Calcula MFCCs (por frame)."""
    mfcc = librosa.feature.mfcc(
        y=data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfcc


def process_folder(root_folder, output_folder):
    """Procesa todo el dataset, crea segmentos y guarda features en .npz."""
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.lower().endswith(".wav"):
                continue

            filepath = os.path.join(dirpath, filename)
            print(f"\n Procesando: {filename}")
            audio, sr = resample(filepath)
            segments = slice_audio(audio, sr)

            for idx, seg in enumerate(segments):
                f0_hz, f0_norm = get_f0(seg, sr)
                loud_db, loud_lin, stft_mag = loudness(seg, sr)
                mfcc = compute_mfcc(seg, sr)

                base_name = f"{os.path.splitext(filename)[0]}_seg{idx}.npz"
                out_path = os.path.join(output_folder, base_name)

                np.savez(
                    out_path,
                    audio=seg,
                    f0_hz=f0_hz,
                    f0_norm=f0_norm,
                    loudness_db_norm=loud_db,
                    loudness_lin=loud_lin,
                    stft_mag=stft_mag,
                    mfcc=mfcc,
                )

                print(f"Segmento {idx+1}: guardado en {base_name}")


process_folder(PATH, OUTPUT_PATH)
