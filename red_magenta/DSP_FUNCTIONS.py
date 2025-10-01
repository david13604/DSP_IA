import tensorflow as tf
import keras
import numpy as np
import librosa
from scipy.signal import lfilter


def loudness(audio, sr, frame_size=1024, hop_size=256):
    """
    Esta es la parte del loudness que calcula A(n)
    """

    S = np.abs(librosa.stft(audio, n_fft=frame_size, hop_length=hop_size))

    P = np.mean(S**2, axis=0)

    loudness_db = librosa.power_to_db(P, ref=np.max)

    return loudness_db

def compute_f0(audio, sr, frame_size= 1024, hop_size=256, fmin=50.0, fmax=20000.0):
    """
    Aca saco f0(n) usando simplemente librosa 
    es muy probable que tenga que cambiar esto a tensorflow(?
    """

    f0, voiced_flag, voice_prob = librosa.pyin(
        audio,
        fmin= fmin,
        fmax= fmax,
        sr= sr,
        frame_length= frame_size,
        hop_length= hop_size
    )

    f0 = np.nan_to_num(f0) #esto transforma los Nan por 0 cuando no suena nada

    return f0

def harmonic_synth(A, c, f_0, f_s, phi0):
    """
    (n) denota que es un vector
    Explicacion mia:
    
    A -> viene del loudness 
    c -> salida de autoencoder pondera a A (n)
    f_0 -> frecuencia fundamenta (n)
    phi_0 -> fase inicial que la dejo en 0 nms
    fs -> fercuencia de muestreo
    """
    
    N,K = c.shape

    if phi0 is None:
        phi0 = np.zeros(K)
    
    phase_f0 = 2 * np.pi * np.cumsum(f_0)/f_s 

    out = np.zeros(N)

    for k in range(1, K+1):
        phase_k = k * phase_f0 + phi0[k-1]
        out += A*c[:, k-1]*np.sin(phase_k) # es probable que tire error aca en el iterable

    return out


def filtered_noise(h, N):
    """
    Esta es la parte del filtro FIR:
    h -> sale de la red es la respuesta al impulso (n)
    N -> cantidad de muestras objetivo
    """

    w = np.random.uniform(-1,1, N + len(h)-1)

    out = lfilter(h, [1.0], w) # (numerador, denominador, señal a filtrar)
    
    return out[:N]