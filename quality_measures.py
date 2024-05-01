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

if __name__ == "__main__":
    image_path = 'Images/Garden/chinese_garden1.png' 
    image = cv2.imread(image_path)
    
    # Calculate quality measures
    contrast, saturation, well_exposedness = calculate_quality_measures(image)
    
    # Display heat maps
    display_heatmaps(image, contrast, saturation, well_exposedness)
