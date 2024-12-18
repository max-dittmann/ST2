import os
import json
import cv2
import numpy as np

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


def load_images_from_folder(folder_path, num_troubleshoot_images, extensions=('jpg', 'png', 'tiff')):
    """
    Loads an adjustable number of images from a folder with given extensions.
    """
    images = []
    i = 0
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(extensions):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Image {filename} not loaded.")
            else:
                images.append(image)
        if i == num_troubleshoot_images:
            break
        i= i + 1

    return images

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

# Function to save parameters as JSON
def save_parameters(file_path, parameters):
    """
    Saves the parameters to a JSON file with the given file path.
    Ensures the directory exists before saving.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directories if they don't exist
    with open(file_path, 'w') as file:
        json.dump(parameters, file, indent=4)
    print(f"Parameters saved to: {file_path}")

def ask_to_save_parameters(input_folder, parameters):
    """
    Asks the user whether they want to save the parameters and saves both apex and COM parameters to JSON files.
    """
    save = input("Would you like to save the parameters? (y/n): ").strip().lower()
    if save == 'y':
        # Save Apex parameters
        parameters_file_name_apex = "p1_apex.json"
        parameters_file_path_apex = os.path.join(input_folder, parameters_file_name_apex)
        save_parameters(parameters_file_path_apex, parameters)

        # Save COM parameters (copy apex parameters but change area_threshold to 500)
        parameters_com = parameters.copy()
        parameters_com["area_threshold"] = 500  # Adjust the area threshold for COM
        parameters_file_name_com = "p1_com.json"
        parameters_file_path_com = os.path.join(input_folder, parameters_file_name_com)
        save_parameters(parameters_file_path_com, parameters_com)
    else:
        print("Parameters not saved.")

# Main function
def main():
    # Define the input folder path for the images
    base_folder = 'C:/Users/maxdi/OneDrive/Desktop/Bachelorthesis/Pycharm/ST2/Photos'
    sub_folder = '13-12-24/6'
    input_folder = os.path.join(base_folder, sub_folder)



    # Define parameters for image processing and troubleshooting
    # Later you will be asked whether you want to save them
    area_threshold = 10             # Apex area threshold
    blur_kernel_size = 3            # Size of kernel when blurring
    binarize_threshold = 150        # Binarization parameter
    troubleshoot = False            # If this is set to True, the pictures of the image processing are printed
    num_troubleshoot_images = 10    # Number of images to troubleshoot

    print("loading images")
    # Load images from the folder
    images = load_images_from_folder(input_folder, num_troubleshoot_images)

    print("images loaded")

    # Process only the first few images for troubleshooting
    troubleshooting_images = images[:num_troubleshoot_images]

    # Call track_apex_centers with troubleshooting for the first few images
    print(f"Processing {num_troubleshoot_images} images in troubleshooting mode...\n")
    apex_coordinates_matrix = track_apex_centers(troubleshooting_images, area_threshold=area_threshold,
                                                 blur_kernel_size=blur_kernel_size,
                                                 binarize_threshold=binarize_threshold, troubleshoot=troubleshoot)

    # Output the tracked apex coordinates for troubleshooting images
    for idx, row in enumerate(apex_coordinates_matrix, start=1):
        print(f"Image {idx}: Apex Coordinates: {row}")

    # Ask if the user would like to save the parameters
    parameters = {
        "area_threshold": area_threshold,
        "blur_kernel_size": blur_kernel_size,
        "binarize_threshold": binarize_threshold,
        "num_troubleshoot_images": num_troubleshoot_images
    }
    ask_to_save_parameters(input_folder, parameters)


if __name__ == "__main__":
    main()
