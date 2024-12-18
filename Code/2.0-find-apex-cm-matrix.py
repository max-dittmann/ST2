import os
import json
import cv2
import numpy as np

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

def preprocess_image(image, blur_kernel_size=7, binarize_threshold=220, troubleshoot=False):
    """
    Preprocesses the image by converting to grayscale, applying Gaussian Blur, and binarizing.
    If troubleshoot is True, it will display the blurred image and the binarized image with the respective titles.
    """
    # Step 1: Convert to grayscale if necessary
    if len(image.shape) == 3:  # Image has 3 channels (likely BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Image is already grayscale, no need to convert

    # Step 2: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    # Step 3: Binarize the image
    _, binarized = cv2.threshold(blurred, binarize_threshold, 255, cv2.THRESH_BINARY)
    binarized = cv2.bitwise_not(binarized)  # Invert the image if needed

    # Step 4: If troubleshoot is True, display the blurred and binarized images
    if troubleshoot:
        # Create the title with the kernel size
        title_blurred = f"Blurred Image (Kernel Size: {blur_kernel_size})"
        # Display the blurred image
        cv2.imshow(title_blurred, blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Display the binarized image with a title
        cv2.imshow("Binarized Image", binarized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return binarized

def find_contours(binarized_image, area_threshold=100, troubleshoot=False):
    """
    Finds contours in the binarized image and filters them based on area.
    If troubleshoot is True, it will display all contours (unfiltered) on a white canvas,
    display the smallest contour that meets the area threshold on a black canvas,
    print the area of the apex contour along with an array of all contour areas (unfiltered),
    and mark the apex center with a blue cross.
    """
    # Step 1: Find all contours in the binarized image
    contours, _ = cv2.findContours(binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Sort all contours by area (smallest to largest)
    sorted_contours = sorted(contours, key=cv2.contourArea)

    # Step 3: Calculate the area of each contour and store in a list (for all contours, unfiltered)
    all_contour_areas = [cv2.contourArea(cnt) for cnt in sorted_contours]

    # Step 4: Filter contours based on the area threshold
    filtered_contours = [cnt for cnt in sorted_contours if cv2.contourArea(cnt) > area_threshold]

    # Step 5: If no valid contours (above threshold) found, return None
    if len(filtered_contours) == 0:
        print(f"No contours found above the threshold of {area_threshold}.")
        return None

    # Step 6: The smallest contour above the threshold is assumed to be the apex contour
    apex_contour = filtered_contours[0]

    # Step 7: If troubleshoot is True, display all contours (unfiltered) and apex contour separately
    if troubleshoot:
        # Get the size of the binarized image
        canvas_size = binarized_image.shape

        # Create a white canvas to display all contours (unfiltered)
        white_canvas = np.ones((canvas_size[0], canvas_size[1]), dtype=np.uint8) * 255  # White background

        # Draw all contours (unfiltered) on the white canvas with black lines
        cv2.drawContours(white_canvas, sorted_contours, -1, (0, 0, 0), 1)  # Draw all contours in black

        # Add text to the white canvas
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(white_canvas, "All contours that were found", (10, 30), font, 0.7, (0, 0, 0), 2)

        # Display the canvas with all contours (unfiltered)
        cv2.imshow('All Contours (Troubleshoot)', white_canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Create a black canvas to display only the smallest contour above the threshold
        black_canvas = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.uint8)  # Black background

        # Draw only the smallest contour (apex contour) on the black canvas with white lines
        cv2.drawContours(black_canvas, [apex_contour], -1, (255, 255, 255), 2)  # White lines

        # Add a label for the smallest contour above the threshold
        cv2.putText(black_canvas, "Smallest Contour Above Threshold", (10, 30), font, 0.7, (255, 255, 255), 2)

        # Display the black canvas with the smallest contour above threshold
        cv2.imshow('Smallest Contour (Troubleshoot)', black_canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print the area of the apex contour (filtered) and the array of all contour areas (unfiltered)
        apex_area = cv2.contourArea(apex_contour)
        print(f"Apex Contour Area: {apex_area:.2f} | All Contour Areas: {all_contour_areas}")

    # Return the apex contour (smallest valid contour above the threshold)
    return apex_contour


def track_apex_centers(images, area_threshold=100, blur_kernel_size=7, binarize_threshold=220, troubleshoot=False):
    """
    Tracks the apex center for each image in the sequence.
    If troubleshoot=True, it will display intermediate steps like blurred images, binarized images, and contours.
    """
    apex_coordinates_matrix = []

    for image in images:
        # Step 1: Preprocess the image (convert to grayscale, blur, binarize)
        binarized_image = preprocess_image(image, blur_kernel_size=blur_kernel_size,
                                           binarize_threshold=binarize_threshold, troubleshoot=troubleshoot)

        # Step 2: Find the apex contour
        apex_contour = find_contours(binarized_image, area_threshold=area_threshold, troubleshoot=troubleshoot)

        if apex_contour is not None:
            # Step 3: Find the center of the apex contour using moments
            M = cv2.moments(apex_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0  # Handle division by zero

            # Step 4: Store the center coordinates
            apex_coordinates_matrix.append([cX, cY])

            # Step 5: If troubleshoot is True, mark the apex center on the original image and display it
            if troubleshoot:
                # Mark the apex in red and center with a blue cross
                cross_size = 10
                cv2.drawContours(image, [apex_contour], -1, (0, 0, 255), 2)
                cv2.line(image, (cX - cross_size, cY), (cX + cross_size, cY), (255, 0, 0), 2)
                cv2.line(image, (cX, cY - cross_size), (cX, cY + cross_size), (255, 0, 0), 2)
                # Display the image with the marked apex
                cv2.imshow('Apex Marked', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return apex_coordinates_matrix

def load_parameters(folder_path, file_name):
    """
    Load parameters from a JSON file.

    Parameters:
    - folder_path: The directory where the JSON file is located.
    - file_name: The name of the JSON file (e.g., p1_apex.json or p1_com.json).

    Returns:
    - parameters: A dictionary of parameters loaded from the file.
    """
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r') as f:
        parameters = json.load(f)

    return parameters

def save_coordinates(folder_path, coordinates_matrix, file_name):
    """
    Save coordinates (apex/COM) as a matrix to a JSON file.

    Parameters:
    - folder_path: The directory where the JSON file will be saved.
    - coordinates_matrix: A matrix (list of lists) of coordinates.
    - file_name: The name of the JSON file (e.g., p1_apex_coords.json or p1_com_coords.json).
    """
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'w') as f:
        json.dump(coordinates_matrix, f, indent=4)

    #print(f"Coordinates saved to: {file_path}")

def main():
    # Define the input folder path for the images
    base_folder = 'C:/Users/maxdi/OneDrive/Desktop/Bachelorthesis/Pycharm/ST2/Photos'
    sub_folder = '13-12-24/6'
    input_folder = os.path.join(base_folder, sub_folder)

    # Load images from the folder
    print("Loading all images from input folder...")
    images = load_images_from_folder(input_folder)
    print("Images loaded")

    # Load apex parameters from p1_apex.json
    apex_parameters = load_parameters(input_folder, file_name="p1_apex.json")
    # Explicitly print the loaded apex parameters
    print("Loaded Apex Parameters:")
    print(f" - Area Threshold: {apex_parameters['area_threshold']}")
    print(f" - Blur Kernel Size: {apex_parameters['blur_kernel_size']}")
    print(f" - Binarize Threshold: {apex_parameters['binarize_threshold']}")

    # Track the apex centers using the loaded parameters
    apex_coordinates_matrix = track_apex_centers(images, area_threshold=apex_parameters["area_threshold"],
                                                 blur_kernel_size=apex_parameters["blur_kernel_size"],
                                                 binarize_threshold=apex_parameters["binarize_threshold"],
                                                 troubleshoot=False)

    # Save the apex coordinates matrix as a JSON file
    save_coordinates(input_folder, apex_coordinates_matrix, file_name="p1_apex_coords.json")
    print(f"Apex coordinates saved to: {input_folder}/p1_apex_coords.json")


    # Load COM parameters from p1_com.json (Area threshold for COM should be different)
    com_parameters = load_parameters(input_folder, file_name="p1_com.json")
    # Explicitly print the loaded COM parameters
    print("Loaded COM Parameters:")
    print(f" - Area Threshold: {com_parameters['area_threshold']}")
    print(f" - Blur Kernel Size: {com_parameters['blur_kernel_size']}")
    print(f" - Binarize Threshold: {com_parameters['binarize_threshold']}")

    # Track the COM centers using the loaded parameters
    com_coordinates_matrix = track_apex_centers(images, area_threshold=com_parameters["area_threshold"],
                                                blur_kernel_size=com_parameters["blur_kernel_size"],
                                                binarize_threshold=com_parameters["binarize_threshold"],
                                                troubleshoot=False)

    # Save the COM coordinates matrix as a JSON file
    save_coordinates(input_folder, com_coordinates_matrix, file_name="p1_com_coords.json")
    print(f"Center of mass coordinates saved to: {input_folder}/p1_com_coords.json")


if __name__ == "__main__":
    main()
