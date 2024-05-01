import cv2
import os

def load_images_from_directory(directory):
    """
    Load all image files from the specified directory.

    Args:
        directory (str): Path to the directory containing image files.

    Returns:
        list: A list of images loaded as numpy arrays.
    """
    # List to store loaded images
    images = []
    # Supported image file extensions
    supported_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an image
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            # Construct full file path
            file_path = os.path.join(directory, filename)
            # Load the image using OpenCV
            image = cv2.imread(file_path)           
            # Check if the image was loaded successfully
            if image is not None:
                images.append(image)
            else:
                print(f"Warning: Failed to load image {file_path}")
        else:
            print(f"Skipped non-image file: {filename}")

    return images

