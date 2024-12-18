

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Define helper functions here...


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


def plot_coordinates_over_time(apex_coordinates_matrix, frame_rate, title="", w_mov=1, w_fix=10, display_duration=None):
    """
    Plots the X and Y coordinates of the apex over time in two subplots, with the time axis calculated from frame rate.
    Adds frame rate, moving window size, and fixed window size at the bottom.
    Optionally restricts the plot to the first 'display_duration' seconds.
    """
    title = ""
    print(f"Printing the X and Y coordinates of the apex over time with w_mov = {w_mov} and w_fix = {w_fix}")

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
    #plt.savefig("apex_coordinates_plot.pdf")  # Save as vector format (PDF)

    # Show the plot
    plt.show()



import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_square_vel(distances_matrix, title="", frame_rate=30, w_mov=1, w_fix=10,
                           display_duration=None):
    """
    Plots the absolute velocities of the apex over time.
    Adds frame rate, moving window size, and fixed window size at the bottom.
    Optionally restricts the plot to the first 'display_duration' seconds.
    """
    title = ""

    # Convert frames to seconds
    total_frames = len(distances_matrix)
    time_steps = np.arange(total_frames) / frame_rate * w_fix  # Divide by frame_rate to get time in seconds


    # Restrict to display_duration if specified
    if display_duration is not None:
        max_frames = int(display_duration * frame_rate / w_fix)
        distances_matrix = distances_matrix[:max_frames]
        time_steps = time_steps[:max_frames]  # Slice time_steps to match display_duration


    # Update plot parameters to match the desired format
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "text.usetex": False,  # Set to True if LaTeX is enabled
    })

    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)

    # Plot velocity squared
    ax.plot(time_steps, distances_matrix, color='b', linestyle='-', label="Absolute Velocity Squared")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Velocity Squared (pixels$^2$/s$^2$)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Set the main title
    # Save the plot

    # Show the plot
    print(f"Plotting velocity squared over time with w_mov = {w_mov} and w_fix = {w_fix}")
    plt.show()




def main():


    # Define the path for the selected JSON file
    base_folder = 'C:/Users/maxdi/OneDrive/Desktop/Bachelorthesis/Pycharm/ST2/Photos'
    sub_folder_com = f'13-12-24/8/p1_com_coords.json'
    #sub_folder2 = f'06-12-24/7/p1_apex_coords.json'
    input_file = os.path.join(base_folder, sub_folder_com)

    # Check if the file exists and load the data
    if os.path.exists(input_file):
        coordinates_matrix = load_apex_coordinates(input_file)
        print(f"Coordinates matrix loaded from: {input_file}")
    else:
        print(f"Error: File {input_file} does not exist.")
        return


    # Define parameters
    w_mov = 1
    w_fix = 10
    delta_t = 1
    frame_rate = 200
    display_duration = 10# CHANGE: Set the duration in seconds for plots

    # Smooth the path
    mov_smooth_matrix = mov_smooth_apex_path(coordinates_matrix, w_mov)
    np_matrix = np.array(mov_smooth_matrix)

    both_smooth_matrix = fix_smooth_apex_path(mov_smooth_matrix, w_fix)

    # Calculate velocities
    velocities_matrix = cent_diff_vel(both_smooth_matrix, delta_t)

    # Plot coordinates over time with updated title

    plot_coordinates_over_time(both_smooth_matrix, frame_rate / w_fix, title="", w_mov=w_mov, w_fix=w_fix, display_duration=display_duration)  # CHANGE: Added display_duration

    # Plot velocities over time with updated title

    # Calculate and print absolute velocity
    velocity_squared = square_vel(velocities_matrix)

    # Plot absolute velocity
    # ATTENTION: This function is behaving super strange. I would just redo the entire thing and hope for the best
    plot_square_vel(
        velocity_squared,
        title=r"Magnitude of Velocity Vector Squared, |v|$^2$ = $(v_x)^2 + (v_y)^2$",  # Ensures LaTeX-style formatting
        frame_rate=frame_rate,
        w_mov=w_mov,
        w_fix=w_fix ,
        display_duration=display_duration  # CHANGE: Added display_duration
    )
if __name__ == "__main__":
    main()
