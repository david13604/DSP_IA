import matplotlib.pyplot as plt  # Add this import at the top of your file
import Peak_finder.Make_database_numpy as Make_database_numpy
import numpy as np

def plot_spectrum_with_peaks(freqs, mag, peak_freqs, peak_amps, title="Spectrum with Peaks"):
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, mag, label="Spectrum")
    plt.scatter(peak_freqs, peak_amps * np.max(mag), color='red', label="Peaks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.legend()
    plt.show()

# Example usage for a single file
test_file = r"C:\Users\usuario\Desktop\DSP_IA_local\DSP_IA\SoundEffects\BluezoneCorp - Steampunk Machines\Bluezone_BC0305_steampunk_machine_mechanical_texture_heavy_impact_011.wav" 
freqs, mag = Make_database_numpy.compute_spectrum(test_file)
peak_freqs, peak_amps = Make_database_numpy.extract_peaks_from_spectrum(mag, freqs, n_peaks=15)
plot_spectrum_with_peaks(freqs, mag, peak_freqs, peak_amps)