import numpy as np
import matplotlib.pyplot as plt
import os
import json

def load_angles_from_json(file_path):
    """Load angles from a JSON file."""
    with open(file_path, 'r') as f:
        angles = json.load(f)
    return angles

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "text.usetex": False,  # Set to True if LaTeX is enabled
})

# Constants and Data Preparation
frame_rate = 200  # Sample rate in Hz
T = 1 / frame_rate  # Sampling interval
filter_threshold = 1  # Threshold for ifft from power spectrum

short_start_time = 0  # start in seconds
short_end_time = 16  # end in seconds

short_start_frame = short_start_time * frame_rate
short_end_frame = short_end_time * frame_rate

base_folder = 'C:/Users/maxdi/OneDrive/Desktop/Bachelorthesis/Pycharm/ST/Photos'
sub_folder = '06-12-24/7'
input_folder = os.path.join(base_folder, sub_folder)

# Define file paths for nu and phi JSON files
nu_file_path = os.path.join(input_folder, 'final_nu_angles.json')
nu_values_original = load_angles_from_json(nu_file_path)  # data type: np array
nu_values_short_original = np.unwrap(np.radians(nu_values_original))

# Select the time window
nu_values_short = nu_values_original[short_start_frame:short_end_frame]

# Step 1: Preprocess Data (Optional)
#nu_detrended = nu_values - np.mean(nu_values)  # Remove the mean
nu_detrended = nu_values_short -np.mean(nu_values_short)
#nu_detrended = nu_values_short


# Step 2: Perform FFT
nu_fft = np.fft.fft(nu_detrended)
frequencies = np.fft.fftfreq(len(nu_values_short), T)

# Step 3: Calculate the Power Spectrum for all frequencies
PSD = np.abs(nu_fft) ** 2  # Power spectrum for all frequencies

# Step 4: Apply the threshold filter and perform inverse FFT
indices = PSD > filter_threshold  # Filter condition with the same shape as nu_fft
nu_fft_filt = indices * nu_fft  # Zero out the low-power frequencies
nu_reconstructed_filt = np.fft.ifft(nu_fft_filt)  # Filtered inverse FFT

# Create subplots with an additional plot for the spectrogram
fig, axs = plt.subplots(1, 1, figsize=(2, 2))  # 4 rows to include the spectrogram

# Convert frames to seconds for the x-axis
time_values_original = np.arange(len(nu_values_original)) * T
time_values_detrended = np.arange(len(nu_detrended)) * T + short_start_time


# Plot 4: Spectrogram
axs.specgram(nu_detrended, NFFT=256, Fs=frame_rate, noverlap=128, detrend="linear", scale= "linear", cmap="binary")  # Spectrogram with detrend
axs.set_xlabel("Time (seconds)")
axs.set_xlim(0, 10)  # Limit

axs.set_ylabel("Frequency (Hz)")
axs.set_ylim(0, 60)  # Limit y-axis to 40 Hz


# Add text below the plots to display fps and subfolder name
#fig.text(0.5, 0.01, f"fps = {frame_rate} | Photo: {sub_folder}", ha='center', fontsize=12)

# Adjust layout for better readability
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Leave space for the fps text at the bottom

# Find the indices of the five largest powers in the PSD array
#top_indices = np.argsort(PSD)[-10:][::-1]  # Sort by power and get the top 5 indices in descending order

# Filter for positive frequencies and their corresponding power values
positive_freqs = frequencies >= 0
positive_frequencies = frequencies[positive_freqs]
positive_PSD = PSD[positive_freqs]

# Find the indices of the 10 largest powers among the positive frequencies
top_indices = np.argsort(positive_PSD)[-10:][::-1]  # Get top 10 indices in descending order

# Calculate the number of positive frequencies
num_positive_frequencies = np.sum(positive_freqs)

# Apply tight_layout with additional padding to prevent overlapping
plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0, rect=[0, 0.03, 1, 0.97])


# Print the number of positive frequencies
print(f"Number of positive frequencies in the frequencies array: {num_positive_frequencies}")

# Print the top 10 positive frequencies and their respective powers
print("Top 10 positive frequencies with the largest power from", short_start_time, "s until", short_end_time, "s :" )
for i, idx in enumerate(top_indices):
    freq = positive_frequencies[idx]
    power = positive_PSD[idx]
    print(f"{i + 1}: Frequency = {freq:.2f} Hz, Power = {power:.2f}")

# Show the plots
plt.show()
