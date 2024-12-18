import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set the frame rate and smoothing window parameters
frame_rate = 200  # frames per second
display_duration = 10 # time in seconds

# Update matplotlib formatting to match the desired style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "text.usetex": False,  # Set to True if LaTeX is enabled
})

def fix_smooth(data, frame_window):
    """Apply fixed window averaging to data."""
    num_points = len(data)
    smoothed_data = []
    for i in range(0, num_points, frame_window):
        window = data[i:i + frame_window]
        if len(window) == frame_window:
            smoothed_data.extend([np.mean(window)] * frame_window)
    return np.array(smoothed_data[:num_points])

def load_angles_from_json(file_path):
    """Load angles from a JSON file."""
    with open(file_path, 'r') as f:
        angles = json.load(f)
    return angles

# Function to compute the derivative of an angle array
def compute_derivative(data, frame_rate):
    """Compute the derivative of the given data array using central difference."""
    derivatives = np.zeros_like(data)
    for i in range(1, len(data) - 1):
        derivatives[i] = (data[i + 1] - data[i - 1]) * frame_rate / 2
    derivatives[0] = (data[1] - data[0]) * frame_rate  # Forward difference at the start
    derivatives[-1] = (data[-1] - data[-2]) * frame_rate  # Backward difference at the end
    return derivatives

def save_and_show_plot(filename, x_label, y_label, title):
    """Save the current plot as a PDF and show it."""
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("", fontsize=8)
    plt.gcf().set_size_inches(5, 2)  # Set figure size
    plt.gcf().tight_layout()
    plt.savefig(f"{filename}.pdf")  # Save as vector format (PDF)
    plt.show()

def plot_nu_over_time(nu_values, frame_rate, display_duration=None):
    """Plot nu angle over time."""
    time_steps = np.arange(len(nu_values)) / frame_rate
    nu_values = np.radians(nu_values)  # Ensure values are in radians

    # Limit data to display_duration if specified
    if display_duration is not None:
        max_frames = int(display_duration * frame_rate)
        time_steps = time_steps[:max_frames]
        nu_values = nu_values[:max_frames]

    # Plot nu over time
    plt.figure()
    plt.plot(time_steps, nu_values, color='blue')
    plt.grid(True, linestyle="--", linewidth=0.5)
    save_and_show_plot("nu_angle_over_time", "Time [s]", r"$\nu$ [radians]", "Nu Angle over Time")

def plot_phi_over_time(phi_values, frame_rate, display_duration=None):
    """Plot phi angle over time with angle unwrapping."""
    time_steps = np.arange(len(phi_values)) / frame_rate

    # Apply angle unwrapping to phi and convert to radians
    phi_values_unwrapped = np.unwrap(np.radians(phi_values))

    # Limit data to display_duration if specified
    if display_duration is not None:
        max_frames = int(display_duration * frame_rate)
        time_steps = time_steps[:max_frames]
        phi_values_unwrapped = phi_values_unwrapped[:max_frames]

    # Plot unwrapped phi over time
    plt.figure()
    plt.plot(time_steps, phi_values_unwrapped, label="", color='red')
    plt.grid(True, linestyle="--", linewidth=0.5)
    save_and_show_plot("phi_angle_over_time", "Time [s]", "φ [radians]", "Phi Angle over Time (Unwrapped)")

def plot_nu_derivative_over_time(nu_derivative, frame_rate, display_duration=None):
    """Plot the derivative of nu angle over time with an average line."""
    time_steps = np.arange(len(nu_derivative)) / frame_rate
    # nu_derivative = np.radians(nu_derivative)  # Convert to radians if needed

    # Limit data to display_duration if specified
    if display_duration is not None:
        max_frames = int(display_duration * frame_rate)
        time_steps = time_steps[:max_frames]
        nu_derivative = nu_derivative[:max_frames]

    # Calculate the average of nu_derivative
    avg_nu_derivative = np.mean(nu_derivative)

    # Plot nu derivative over time
    plt.figure()
    plt.plot(time_steps, nu_derivative, label="Nu Derivative", color='blue')
    plt.axhline(avg_nu_derivative, color='lightblue', linestyle='--', label=f"Average: {avg_nu_derivative:.2f} radians/s")
    plt.grid(True, linestyle="--", linewidth=0.5)
    save_and_show_plot("nu_derivative_over_time", "Time [s]", r"(d$\nu$/dt) [radians/s]", "Nu Derivative over Time")

def plot_phi_derivative_over_time(phi_derivative, frame_rate, display_duration=None):
    """Plot the derivative of phi angle over time with an average line."""
    time_steps = np.arange(len(phi_derivative)) / frame_rate
    # phi_derivative = np.radians(phi_derivative)  # Convert to radians if needed

    # Limit data to display_duration if specified
    if display_duration is not None:
        max_frames = int(display_duration * frame_rate)
        time_steps = time_steps[:max_frames]
        phi_derivative = phi_derivative[:max_frames]

    # Calculate the average of phi_derivative
    avg_phi_derivative = np.mean(phi_derivative)

    # Plot phi derivative over time
    plt.figure()
    plt.plot(time_steps, phi_derivative, label="Phi Derivative", color='red')
    plt.axhline(avg_phi_derivative, color='lightcoral', linestyle='--', label=f"Average: {avg_phi_derivative:.2f} radians/s")
    plt.grid(True, linestyle="--", linewidth=0.5)
    save_and_show_plot("phi_derivative_over_time", "Time [s]", r"(d$\phi$/dt) [radians/s]", "Phi Derivative over Time")

def main():
    base_folder = 'C:/Users/maxdi/OneDrive/Desktop/Bachelorthesis/Pycharm/ST2/Photos'
    sub_folder = '13-12-24/8'
    input_folder = os.path.join(base_folder, sub_folder)

    # Define file paths for nu and phi JSON files
    nu_file_path = os.path.join(input_folder, 'final_nu_angles.json')
    phi_file_path = os.path.join(input_folder, 'final_phi_angles.json')

    # Load nu and phi values
    nu_values = load_angles_from_json(nu_file_path) # in degrees
    phi_values = load_angles_from_json(phi_file_path) # in degrees

    # The angles are stored in degrees, so they still have to be converted to radians
    nu_values_rad = np.radians(nu_values)
    phi_values_unwrapped_rad = np.unwrap(np.radians(phi_values))

    # Compute derivatives for nu and phi
    nu_derivative = compute_derivative(nu_values_rad, frame_rate)
    phi_derivative = compute_derivative(phi_values_unwrapped_rad, frame_rate)


    # Plot each graph individually
    print(f"Display duration = {display_duration} seconds")
    print("Plotting nu values")
    plot_nu_over_time(nu_values, frame_rate, display_duration)
    print("Plotting phi values")
    plot_phi_over_time(phi_values, frame_rate, display_duration)
    print("Plotting nu derivatives with average marked")
    plot_nu_derivative_over_time(nu_derivative, frame_rate, display_duration)
    print("Plotting phi derivatives with average marked")
    plot_phi_derivative_over_time(phi_derivative, frame_rate, display_duration)

if __name__ == "__main__":
    main()
