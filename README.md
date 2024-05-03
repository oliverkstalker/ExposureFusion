CSCE 448 Final Project Report - Exposure Fusion
Diego Lanz - 431005222
Oliver Stalker - 331001655
5/2/2024

### Project Overview
This project develops an exposure fusion system based on the paper of the same name (Mertens et. al.) that combines multiple differently exposed images of the same scene into a single image with enhanced visual quality and dynamic range. Unlike traditional high dynamic range (HDR) imaging, which requires complex processing and tone mapping, exposure fusion provides a straightforward and efficient alternative. It directly merges the best-exposed parts of each input image, resulting in a final image that displays detailed highlights and shadows without the need for additional HDR processing. This method is particularly beneficial for capturing scenes with extreme lighting variations where conventional photography might fail to capture the full range of light and detail.

### Implementation
Our implementation follows the techniques described in the “Exposure Fusion” paper by Mertens et. al. and consists of several key steps:

__Image Loading and Weight Calculation:__
Images are loaded from an input directory in order of increasing exposure, and for each 
Exposure a weight map is calculated for the image. These weight maps assess the quality of each image based on contrast, saturation, and well-exposedness. Higher weights are given to pixels that are better exposed, more colorful, and have higher contrast.
__Pyramid Construction:__
For each image & weight map in the corresponding exposure stack, two pyramids are constructed. For the image, a Laplacian pyramid is constructed to preserve the image details at various scales. For the weight map a gaussian pyramid that helps with smooth blending of the pyramids. 
__Pyramid Blending:__
The pyramids are then blended using the weight maps. At each level of the pyramid, the images are additively combined based on the normalized weights from the Gaussian pyramid. This blending process ensures that pixels with higher quality metrics contribute more to the final image.
__Pyramid Collapsing:__
This step involves upsampling and combining the images from the finest to coarsest level of the pyramid, allowing for a final resulting image to be produced. The final image is then rescaled to ensure that it is in a displayable range.


#### File Structure:
weight_map.py: calculates the weight map for each image
image_loader.py: loads the images from a directory
exposure_fusion: applies exposure fusion algorithm










