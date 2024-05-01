import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_quality_measures(image):
    """
    Calculate the quality measures (contrast, saturation, and well-exposedness) for an image.
    Args:
        image (numpy.ndarray): Input image in BGR format.
    Returns:
        tuple: Three numpy arrays representing weight maps for contrast, saturation, and well-exposedness.
    """
    # Ensure the image is in floating point format
    image = np.float32(image) / 255.0
    
    # Convert to grayscale for contrast calculation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate contrast using the Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    contrast = np.abs(laplacian)
    
    # Calculate saturation as the standard deviation of color channels
    mean = np.mean(image, axis=2)
    saturation = np.sqrt(((image - mean[:, :, np.newaxis])**2).mean(axis=2))
    
    # Calculate well-exposedness using a Gaussian curve centered at 0.5
    sigma = 0.2
    well_exposedness = np.exp(-0.5 * ((gray - 0.5) ** 2) / sigma**2)
    
    return contrast, saturation, well_exposedness

def display_heatmaps(image, contrast, saturation, well_exposedness):
    """
    Display heat maps for contrast, saturation, and well-exposedness.
    Args:
        image (numpy.ndarray): Original image.
        contrast, saturation, well_exposedness (numpy.ndarray): Quality measures.
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(contrast, cmap='hot')
    plt.title('Contrast Heatmap')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(saturation, cmap='hot')
    plt.title('Saturation Heatmap')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(well_exposedness, cmap='hot')
    plt.title('Well-Exposedness Heatmap')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_weight_map(image, wc=1.0, ws=1.0, we=1.0):
    """
    Calculate the final weight map for an image using the quality measures.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        wc (float): Exponent for the contrast weight.
        ws (float): Exponent for the saturation weight.
        we (float): Exponent for the well-exposedness weight.
        
    Returns:
        numpy.ndarray: The final weight map.
    """
    # Calculate the quality measures
    contrast, saturation, well_exposedness = calculate_quality_measures(image)
    
    # Compute the weight map using the given exponents
    weight_map = (contrast ** wc) * (saturation ** ws) * (well_exposedness ** we)
    
    # It's common to normalize the weight map to prevent numerical instability
    weight_map += 1e-12  # Prevent division by zero
    
    weight_sum = np.sum(weight_map, axis=(0, 1), keepdims=True) + 1e-12
    weight_map /= weight_sum

    return weight_map

def display_weight_map(image, weight_map):
    """
    Display the weight map for an image.
    
    Args:
        image (numpy.ndarray): Original image.
        weight_map (numpy.ndarray): Final weight map.
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(weight_map, cmap='hot')
    plt.title('Weight Map')
    plt.colorbar()
    plt.axis('off')
    
    plt.show()


if __name__ == "__main__":
    image_path = 'Images/Venice/venice_under.png' 
    image = cv2.imread(image_path)
    
    # Calculate quality measures
    contrast, saturation, well_exposedness = calculate_quality_measures(image)
    
    # Display heat maps
    display_heatmaps(image, contrast, saturation, well_exposedness)

    # Calculate and display weight map
    weight_map = calculate_weight_map(image)

    # Display weight map
    display_weight_map(image, weight_map)
