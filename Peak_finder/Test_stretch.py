import matplotlib.pyplot as plt
from Make_database import (
    compute_spectrum,
    stretch_spectrum,
    extract_peaks_from_spectrum,
)
import numpy as np

#audio_path = r"C:\Users\usuario\Desktop\DSP_IA_local\DSP_IA\SoundEffects\BluezoneCorp - Steampunk Machines\Bluezone_BC0305_steampunk_machine_mechanical_texture_heavy_impact_011.wav" 
audio_path = "/mnt/c/Users/matth/OneDrive/Desktop/PUC/DSP_IA/SoundEffects/BluezoneCorp - Steampunk Machines/Bluezone_BC0305_steampunk_machine_mechanical_texture_heavy_impact_011.wav"
stretch_factor = 2
n_peaks = 15

# Original spectrum
freqs, mag = compute_spectrum(audio_path)
norm_mag = mag / np.max(mag + 1e-12)
peak_freqs, peak_amps = extract_peaks_from_spectrum(mag, freqs, n_peaks)

# Stretched spectrum
mag_stretched = stretch_spectrum(mag, stretch_factor)
norm_mag_stretched = mag_stretched / np.max(mag_stretched + 1e-12)
peak_freqs_st, peak_amps_st = extract_peaks_from_spectrum(mag_stretched, freqs, n_peaks)

plt.figure(figsize=(12, 6))

# Plot original
plt.subplot(2, 1, 1)
plt.plot(freqs, norm_mag, label="Original Spectrum")
plt.scatter(peak_freqs, peak_amps, color="red", label="Peaks")
plt.title("Original Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")
plt.legend()

# Plot stretched
plt.subplot(2, 1, 2)
plt.plot(
    freqs, norm_mag_stretched, label=f"Stretched Spectrum (factor={stretch_factor})"
)
plt.scatter(peak_freqs_st, peak_amps_st, color="red", label="Peaks")
plt.title("Stretched Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")
plt.legend()

plt.tight_layout()
plt.show()
