import matplotlib.pyplot as plt
from Make_database import compute_spectrum, shift_spectrum, extract_peaks_from_spectrum
import numpy as np


audio_path = r"C:\Users\usuario\Desktop\DSP_IA_local\DSP_IA\SoundEffects\BluezoneCorp - Steampunk Machines\Bluezone_BC0305_steampunk_machine_mechanical_texture_heavy_impact_011.wav" 
shift_bins = 5000  # Try positive or negative values
n_peaks = 15

# Original spectrum
freqs, mag = compute_spectrum(audio_path)
norm_mag = mag / np.max(mag + 1e-12)
peak_freqs, peak_amps = extract_peaks_from_spectrum(mag, freqs, n_peaks)

# Shifted spectrum
mag_shifted = shift_spectrum(mag, shift_bins)
norm_mag_shifted = mag_shifted / np.max(mag_shifted + 1e-12)
peak_freqs_sh, peak_amps_sh = extract_peaks_from_spectrum(mag_shifted, freqs, n_peaks)

plt.figure(figsize=(12, 6))

# Plot original
plt.subplot(2, 1, 1)
plt.plot(freqs, norm_mag, label="Original Spectrum")
plt.scatter(peak_freqs, peak_amps, color="red", label="Peaks")
plt.title("Original Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")
plt.legend()

# Plot shifted
plt.subplot(2, 1, 2)
plt.plot(freqs, norm_mag_shifted, label=f"Shifted Spectrum (bins={shift_bins})")
plt.scatter(peak_freqs_sh, peak_amps_sh, color="red", label="Peaks")
plt.title("Shifted Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")
plt.legend()

plt.tight_layout() 
plt.show()