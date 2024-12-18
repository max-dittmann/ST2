import os
import json
import numpy as np
#from src2.absolute_value import abs_val
from src2.apex_derivative import cent_diff_vel
import cv2
import matplotlib.pyplot as plt

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Define helper functions here...

def mov_smooth_apex_path(apex_coordinates_matrix, window_size):
    """
    Smooths the apex path using a sliding window average.
    Parameters:
    - apex_coordinates_matrix: List of (x, y) positions for the apex.
    - window_size: The size of the window used for smoothing.
    Returns:
    - moving_average_apex_matrix: List of smoothed (x, y) positions.
    """
    moving_average_apex_matrix = []
    num_points = len(apex_coordinates_matrix)

    for i in range(num_points):
        # Define the window range (start and end)
        start = max(0, i - window_size + 1)  # Start of the sliding window
        end = i + 1  # End of the sliding window (inclusive)

        # Get the points within the sliding window
        window_points = apex_coordinates_matrix[start:end]

        # Compute the average of the points in the window
        avg_x = int(np.mean([point[0] for point in window_points]))
        avg_y = int(np.mean([point[1] for point in window_points]))

        # Append the smoothed point to the list
        moving_average_apex_matrix.append([avg_x, avg_y])

    return moving_average_apex_matrix


def fix_smooth_apex_path(apex_coordinates_matrix, frame_window):
    """
    Averages the apex path over a fixed number of frames.
    Parameters:
    - apex_coordinates_matrix: List of (x, y) apex positions.
    - frame_window: The number of frames used to average.
    Returns:
    - fixed_window_average_apex_matrix: List of averaged (x, y) positions.
    """
    fixed_window_average_apex_matrix = []
    num_points = len(apex_coordinates_matrix)

    for i in range(0, num_points, frame_window):
        # Get the points in the current frame window
        window_points = apex_coordinates_matrix[i:i + frame_window]

        # Check if the window has enough points
        if len(window_points) == frame_window:
            # Compute the average position
            avg_x = int(np.mean([point[0] for point in window_points]))
            avg_y = int(np.mean([point[1] for point in window_points]))

            # Append the averaged position
            fixed_window_average_apex_matrix.append([avg_x, avg_y])

    return fixed_window_average_apex_matrix

def load_apex_coordinates(file_path):
    """
    Load apex coordinates from a JSON file.

    Parameters:
    - file_path: The path to the JSON file containing the apex coordinates.

    Returns:
    - apex_coords_matrix: A matrix (list of lists) of apex coordinates.
    """
    with open(file_path, 'r') as f:
        apex_coords_matrix = json.load(f)

    return np.array(apex_coords_matrix)

def square_vel(apex_coordinates_matrix):
    """
    Calculate the distance between consecutive apex points.

    Parameters:
    - apex_coordinates_matrix: List of (x, y) coordinates of the apex.

    Returns:
    - distances_matrix: List of distances between consecutive apex points.
    """
    distances_matrix = []

    for i in range(len(apex_coordinates_matrix)):
        # Get the coordinates of two consecutive points
        x1, y1 = apex_coordinates_matrix[i]

        # Calculate the distance using the Pythagorean theorem
        distance = ((x1) ** 2 + (y1) ** 2)

        # Save the distance
        distances_matrix.append(distance)

        # Print for troubleshooting
        #print(f"Point 1: ({x1}, {y1}) -> Point 2: ({x2}, {y2}) -> Distance: {distance:.2f}")

    return distances_matrix

def cent_diff_vel(apex_coordinates_matrix, delta_t):
    """
    Calculates the velocity (derivatives of x and y) of the apex using the central difference scheme.

    Parameters:
    - apex_coordinates_matrix: Matrix with two columns representing the (x, y) coordinates of the apex.
    - delta_t: Time step between consecutive frames.

    Returns:
    - velocity_matrix: Matrix with two columns representing the derivatives of x and y over time.
    """
    num_points = len(apex_coordinates_matrix)
    velocity_matrix = np.zeros((num_points, 2))  # Initialize velocity matrix with same number of rows as input

    # For the first point (use forward difference)
    velocity_matrix[0, 0] = (apex_coordinates_matrix[1][0] - apex_coordinates_matrix[0][0]) / delta_t  # dx/dt
    velocity_matrix[0, 1] = (apex_coordinates_matrix[1][1] - apex_coordinates_matrix[0][1]) / delta_t  # dy/dt

    # For interior points (use central difference)
    for i in range(1, num_points - 1):
        velocity_matrix[i, 0] = (apex_coordinates_matrix[i + 1][0] - apex_coordinates_matrix[i - 1][0]) / (
                    2 * delta_t)  # dx/dt
        velocity_matrix[i, 1] = (apex_coordinates_matrix[i + 1][1] - apex_coordinates_matrix[i - 1][1]) / (
                    2 * delta_t)  # dy/dt

    # For the last point (use backward difference)
    velocity_matrix[num_points - 1, 0] = (apex_coordinates_matrix[num_points - 1][0] -
                                          apex_coordinates_matrix[num_points - 2][0]) / delta_t  # dx/dt
    velocity_matrix[num_points - 1, 1] = (apex_coordinates_matrix[num_points - 1][1] -
                                          apex_coordinates_matrix[num_points - 2][1]) / delta_t  # dy/dt

    return velocity_matrix









# Function definitions...

import matplotlib.pyplot as plt

def plot_apex_coordinates_over_time(apex_coordinates_matrix, frame_rate, title="", w_mov=1, w_fix=10, display_duration=None):
    """
    Plots the X and Y coordinates of the apex over time in two subplots, with the time axis calculated from frame rate.
    Adds frame rate, moving window size, and fixed window size at the bottom.
    Optionally restricts the plot to the first 'display_duration' seconds.
    """
    title = ""
    print("The frame rate in coors plot is equal to: ", frame_rate)

    # Extract X and Y coordinates
    x_coords = [coord[0] for coord in apex_coordinates_matrix]
    y_coords = [coord[1] for coord in apex_coordinates_matrix]
    total_frames = len(apex_coordinates_matrix)
    time_steps = [(i / frame_rate) for i in range(total_frames)]

    # Restrict to display_duration if specified
    if display_duration is not None:
        max_frames = int(display_duration * frame_rate)
        x_coords = x_coords[:max_frames]
        y_coords = y_coords[:max_frames]
        time_steps = time_steps[:max_frames]

    # Update plot parameters to match the desired format
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "text.usetex": False,  # Set to True if LaTeX is enabled
    })

    # Create the plot
    fig, axs = plt.subplots(2, 1, figsize=(5, 3), constrained_layout=True)

    # X coordinates over time
    axs[0].plot(time_steps, x_coords, color='r', linestyle='-', label="X Coordinates")
    axs[0].set_ylabel("X [pixels]")
    axs[0].grid(True)

    # Y coordinates over time
    axs[1].plot(time_steps, y_coords, color='b', linestyle='-', label="Y Coordinates")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Y [pixels]")
    axs[1].grid(True)

    # Set the main title
    fig.suptitle(title, fontsize=12)

    # Save the plot
    plt.savefig("apex_coordinates_plot.pdf")  # Save as vector format (PDF)

    # Show the plot
    plt.show()



def plot_square_vel(distances_matrix, title="Log-Log Plot of Velocity Squared Over Time", frame_rate=30, w_mov=1, w_fix=10,
                           display_duration=None):
    """
    Plots the absolute velocities of the apex over time on a log-log scale.
    Adds frame rate, moving window size, and fixed window size at the bottom.
    Optionally restricts the plot to the first 'display_duration' seconds.
    """
    title = ""

    # Convert frames to seconds
    total_frames = len(distances_matrix)
    time_steps = np.arange(total_frames) / frame_rate * w_fix  # Divide by frame_rate to get time in seconds

    distances_matrix = np.array(distances_matrix)
    dimensions = distances_matrix.shape
    print("Dimensions of the distances square velocity matrix:", dimensions)

    # Restrict to display_duration if specified
    if display_duration is not None:
        max_frames = int(display_duration * frame_rate / w_fix)
        print("maxframes:", max_frames)
        distances_matrix = distances_matrix[:max_frames]
        time_steps = time_steps[:max_frames]  # Slice time_steps to match display_duration

    dimensions = distances_matrix.shape
    print("Dimensions of the distances square velocity matrix after slicing:", dimensions)

    # Update plot parameters to match the desired format
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "text.usetex": False,  # Set to True if LaTeX is enabled
    })

    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)

    # Plot velocity squared on a log-log scale
    ax.loglog(time_steps, distances_matrix, color='b', linestyle='-', label="Absolute Velocity Squared")
    ax.set_xlabel("Time [s] (log scale)")
    ax.set_ylabel(r"Velocity Squared (pixels$^2$/s$^2$) (log scale)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Set the main title
    fig.suptitle(title, fontsize=12)

    # Add frame rate, moving window size, and fixed window size at the bottom of the plot


    # Save the plot
    plt.savefig("velocity_squared_loglog_plot.pdf")  # Save as vector format (PDF)

    # Show the plot
    plt.show()




def main():
    # Define the paths for the selected JSON files
    base_folder = 'C:/Users/maxdi/OneDrive/Desktop/Bachelorthesis/Pycharm/ST2/Photos'
    sub_folder1 = f'13-12-24/8/p1_com_coords.json'
    sub_folder2 = f'13-12-24/6/p1_com_coords.json'
    sub_folder3 = f'06-12-24/7/p1_com_coords.json'
    sub_folder4 = f'06-12-24/6/p1_com_coords.json'
    input_files = [
        os.path.join(base_folder, sub_folder1),
        os.path.join(base_folder, sub_folder2),
        os.path.join(base_folder, sub_folder3),
        os.path.join(base_folder, sub_folder4)
    ]

    # Initialize data storage
    all_coordinates = []
    labels = ["Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4"]
    colors = ['r', 'b', 'g', 'm']  # Assign different colors for each dataset

    # Check if the files exist and load the data
    for file_path in input_files:
        if os.path.exists(file_path):
            coordinates_matrix = load_apex_coordinates(file_path)
            print(f"Coordinates matrix loaded from: {file_path}")
            all_coordinates.append(coordinates_matrix)
        else:
            print(f"Error: File {file_path} does not exist.")
            return

    # Define parameters
    w_mov = 1
    w_fix = 10
    delta_t = 1
    frame_rate = 200
    display_duration = 10  # Restrict plots to 10 seconds

    # Smooth paths and calculate velocities
    smoothed_coordinates = []
    velocities_squared = []
    for coordinates_matrix in all_coordinates:
        # Smooth the path
        mov_smooth_matrix = mov_smooth_apex_path(coordinates_matrix, w_mov)
        both_smooth_matrix = fix_smooth_apex_path(mov_smooth_matrix, w_fix)
        smoothed_coordinates.append(both_smooth_matrix)

        # Calculate velocities
        velocities_matrix = cent_diff_vel(both_smooth_matrix, delta_t)
        velocity_squared = square_vel(velocities_matrix)
        velocities_squared.append(velocity_squared)

    # Plot coordinates over time
    plt.figure(figsize=(5, 3), constrained_layout=True)
    for i, coords in enumerate(smoothed_coordinates):
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        time_steps = [(j / (frame_rate / w_fix)) for j in range(len(coords))]
        time_steps = [t for t in time_steps if t <= display_duration]  # Restrict to 10 seconds
        x_coords = x_coords[:len(time_steps)]
        y_coords = y_coords[:len(time_steps)]

        plt.plot(time_steps, x_coords, color=colors[i], linestyle='-', label=f"X {labels[i]}")
        plt.plot(time_steps, y_coords, color=colors[i], linestyle='--', label=f"Y {labels[i]}")

    plt.xlabel("Time [s]")
    plt.ylabel("Coordinates [pixels]")
    plt.legend()
    plt.grid(True)
    plt.title("Apex Coordinates Over Time")
    plt.savefig("apex_coordinates_combined_plot.pdf")
    plt.show()

    # Plot velocity squared on a log-log scale
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)

    for i, vel_squared in enumerate(velocities_squared):
        # Calculate time steps for each dataset
        time_steps = np.arange(len(vel_squared)) / (frame_rate / w_fix)
        time_steps = time_steps[time_steps <= display_duration]  # Restrict to 10 seconds
        vel_squared = vel_squared[:len(time_steps)]

        # Plot velocity squared on a log-log scale for each dataset
        ax.loglog(time_steps, vel_squared, color=colors[i], linestyle='-', label=labels[i])

    # Set axis labels and title
    ax.set_xlabel("Time [s] (log scale)")
    ax.set_ylabel(r"Velocity Squared (pixels$^2$/s$^2$) (log scale)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.suptitle("Log-Log Plot of Velocity Squared Over Time", fontsize=12)

    # Save the log-log plot
    plt.savefig("velocity_squared_combined_loglog_plot.pdf")
    plt.show()

    # Plot velocity squared on a linear scale
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)

    for i, vel_squared in enumerate(velocities_squared):
        # Calculate time steps for each dataset
        time_steps = np.arange(len(vel_squared)) / (frame_rate / w_fix)
        time_steps = time_steps[time_steps <= display_duration]  # Restrict to 10 seconds
        vel_squared = vel_squared[:len(time_steps)]

        # Plot velocity squared on a linear scale for each dataset
        ax.plot(time_steps, vel_squared, color=colors[i], linestyle='-', label=labels[i])

    # Set axis labels and title
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Velocity Squared (pixels$^2$/s$^2$)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    # Save the linear plot
    plt.savefig("velocity_squared_combined_linear_plot.pdf")
    plt.show()


if __name__ == "__main__":
    main()
