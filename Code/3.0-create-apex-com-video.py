# This creates a video of the apex and cm coordinates trajecories
# The data is displayed on top of the  original footage

import os
import json
import cv2

def load_images_from_folder(folder_path, extensions=('jpg', 'png', 'tiff')):
    """
    Loads all images from a folder with given extensions.
    """
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(extensions):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Image {filename} not loaded.")
            else:
                images.append(image)
    return images

def create_apex_trajectory_video(input_folder, output_video_path, apex_coordinates_matrix, com_coordinates_matrix, video_fps=30):
    """
    Creates a video of images from the input folder, marking the apex and COM positions and drawing their trajectories.
    Displays the frame number in the top left corner of each frame.

    Parameters:
    - input_folder: Path to the folder containing images.
    - output_video_path: Path to save the generated video.
    - apex_coordinates_matrix: List of (x, y) apex coordinates for each image.
    - com_coordinates_matrix: List of (x, y) COM coordinates for each image.
    - video_fps: Frames per second for the video output.
    """
    # Load the image sequence
    images = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(('.jpg', '.png', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)

    # Check if images were loaded
    if not images:
        print("No images found in the input folder.")
        return

    # Get image dimensions (assume all images are the same size)
    height, width, _ = images[0].shape

    # Create VideoWriter to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (width, height))

    # Loop over each frame, mark the apex, COM, their trajectories, and display the frame number
    for i, image in enumerate(images):
        # Clone the image to mark the trajectory and apex
        frame = image.copy()

        # Draw the apex trajectory up to the current frame
        for j in range(1, i + 1):
            prev_apex_point = apex_coordinates_matrix[j - 1]
            curr_apex_point = apex_coordinates_matrix[j]
            cv2.line(frame, (prev_apex_point[0], prev_apex_point[1]), (curr_apex_point[0], curr_apex_point[1]), (255, 0, 0), 2)  # Blue line

        # Draw the COM trajectory up to the current frame  # ***NEW: Draw COM trajectory***
        for j in range(1, i + 1):
            prev_com_point = com_coordinates_matrix[j - 1]
            curr_com_point = com_coordinates_matrix[j]
            cv2.line(frame, (prev_com_point[0], prev_com_point[1]), (curr_com_point[0], curr_com_point[1]), (0, 255, 0), 2)  # Green line

        # Draw the current apex point
        apex_point = apex_coordinates_matrix[i]
        cv2.circle(frame, (apex_point[0], apex_point[1]), 5, (0, 0, 255), -1)  # Red dot for the current apex

        # Draw the current COM point  # ***NEW: Draw COM point***
        com_point = com_coordinates_matrix[i]
        cv2.circle(frame, (com_point[0], com_point[1]), 5, (0, 165, 255), -1)  # Orange dot for the current COM

        # Display the frame number in the top left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)  # Black color for the text
        font_thickness = 2
        text_position = (10, 30)  # Position the text at (10, 30) from the top-left corner
        cv2.putText(frame, f'Frame: {i + 1}', text_position, font, font_scale, font_color, font_thickness)

        # Write the frame with markings to the video
        out.write(frame)

    # Release the video writer
    out.release()

    print(f"Video saved to {output_video_path}")

# Function to load coordinates from JSON file  # ***MODIFIED: Generic function for loading both apex and COM***
def load_coordinates(file_path):
    """
    Load coordinates from a JSON file.

    Parameters:
    - file_path: The path to the JSON file containing the coordinates.

    Returns:
    - coords_matrix: A matrix (list of lists) of coordinates.
    """
    with open(file_path, 'r') as f:
        coords_matrix = json.load(f)

    return coords_matrix

def main():
    # Define the input folder path
    input_folder = 'C:/Users/maxdi/OneDrive/Desktop/Bachelorthesis/Pycharm/ST2/Photos/13-12-24/7'  # Folder containing the sequence of images
    apex_coords_file = 'p1_apex_coords.json'  # JSON file for apex coordinates
    com_coords_file = 'p1_com_coords.json'  # JSON file for COM coordinates
    output_video_path = 'C:/Users/maxdi/OneDrive/Desktop/Bachelorthesis/Pycharm/ST2/data/13-12-24/7/trajectory_video_test.mp4'

    # Set video parameters
    video_fps = 200  # Frames per second for video

    # Load images from the folder
    #images = load_images_from_folder(input_folder)

    # Load the apex and COM coordinates matrices from the JSON files  # ***MODIFIED: Load both apex and COM coordinates***
    apex_coordinates_matrix_path = os.path.join(input_folder, apex_coords_file)
    apex_coordinates_matrix = load_coordinates(apex_coordinates_matrix_path)

    com_coordinates_matrix_path = os.path.join(input_folder, com_coords_file)
    com_coordinates_matrix = load_coordinates(com_coordinates_matrix_path)

    # Create a video of the apex and COM trajectories using the loaded coordinates
    print("Creating video of apex and COM trajectories...")

    # Create the video
    create_apex_trajectory_video(
        input_folder,
        output_video_path,
        apex_coordinates_matrix,
        com_coordinates_matrix,  # ***NEW: Pass COM coordinates matrix to the function***
        video_fps=video_fps
    )

    print("Video creation complete!")


if __name__ == "__main__":
    main()