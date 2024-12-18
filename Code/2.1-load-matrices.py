# This script is meant for troubleshooting
# It prints the apex and center of mass coordinates for each image

import os
import json






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

    return apex_coords_matrix


# Main function
def main():
    # Define the input folder path and the JSON file name
    base_folder = 'C:/Users/maxdi/OneDrive/Desktop/Bachelorthesis/Pycharm/ST2/Photos'
    sub_folder = '13-12-24/8'
    input_folder = os.path.join(base_folder, sub_folder)
    apex_file_name = 'p1_apex_coords.json'
    com_file_name = 'p1_com_coords.json'

    # Create the full file path to load the JSON file
    apex_file_path = os.path.join(input_folder, apex_file_name)
    com_file_path = os.path.join(input_folder, com_file_name)

    # Load the apex coordinates matrix from the JSON file
    apex_coords_matrix = load_apex_coordinates(apex_file_path)

    com_coords_matrix = load_apex_coordinates(com_file_path)



    # Print the loaded apex and center of mass coordinates
    print(f"Loaded apex coordinates from {apex_file_path} and center of mass coordinates from {com_file_path}:\n")
    for idx, (row, row2) in enumerate(zip(apex_coords_matrix, com_coords_matrix), start=1):
        print(f"Image {idx}: Apex Coordinates: {row}, Center of Mass Coordinates: {row2}")




if __name__ == "__main__":
    main()