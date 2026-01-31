import numpy as np
from scipy.signal import firwin, kaiserord, freqz
import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt

def reverse_half(x):
    mask = torch.ones_like(x)
    mask[..., 1::2, ::2] = -1
    return x * mask


def kaiser_filter(wc, atten, N=None):
    """
    wc : frecuencia de corte angular (rad/sample)
    atten : atenuación stopband (dB)
    """
    N_, beta = kaiserord(atten, wc / np.pi)
    N_ = 2 * ((N_ // 2) + 1)  # impar
    N = N if N is not None else N_
    h = firwin(N, wc / np.pi, window=("kaiser", beta), scale=False)
    return h


def get_filter_banks(h, M=16):
    """
    Banco QMF por modulación coseno.
    Importante: n debe tener la MISMA longitud que h
    """
    h = h.view(1, -1)                 # (1, N)
    N = h.shape[-1]

    n = torch.arange(N) - (N - 1) / 2 # centrado, longitud N
    k = torch.arange(M).view(-1, 1)   # (M, 1)
    phi = ((-1) ** k) * np.pi / 4

    mod = torch.cos((2 * k + 1) * np.pi / (2 * M) * n + phi)
    hk = 2 * h * mod                  # (M, N)
    return hk


def polyphase_forward(x, hk, rearrange_filter=True):
    M = hk.shape[0]
    x = rearrange(x, "b c (t m) -> b (c m) t", m=M)

    if rearrange_filter:
        hk = rearrange(hk, "c (t m) -> c m t", m=M)

    x = nn.functional.conv1d(x, hk, padding=hk.shape[-1] // 2)[..., :-1]
    return x


class PQMF(nn.Module):
    def __init__(self, M=16, atten=100):
        super().__init__()
        wc = np.pi / M
        h = kaiser_filter(wc, atten)
        h = torch.from_numpy(h).float()
        hk = get_filter_banks(h, M=M)

        self.register_buffer("h", h)
        self.register_buffer("hk", hk)
        self.M = M

    def forward(self, x):
        x = polyphase_forward(x, self.hk)
        x = reverse_half(x)
        return x

if __name__ == "__main__":

    fs = 48000
    M = 16
    atten = 100

    pqmf = PQMF(M=M, atten=atten)

    plt.figure(figsize=(10, 5))

    for k in range(M):
        hk = pqmf.hk[k].cpu().numpy()
        w, H = freqz(hk, worN=4096)
        f = w / np.pi * (fs / 2)
        plt.plot(f, 20 * np.log10(np.abs(H) + 1e-8))

    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (dB)")
    plt.title("Banco PQMF (16 bandas)")
    plt.ylim(-120, 5)
    plt.xlim(0, fs / 2)
    plt.grid(True)
    plt.show()
