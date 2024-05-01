import cv2
import numpy as np
import matplotlib.pyplot as plt
from weight_map import calculate_weight_map
from image_loader import load_images_from_directory 

def build_pyramid(image, max_level, pyramid_type='laplacian'):
    """ Build Gaussian or Laplacian pyramid for the given image. """
    g_pyramid = [image]
    for i in range(max_level):
        image = cv2.pyrDown(image)
        g_pyramid.append(image)
    
    if pyramid_type == 'gaussian':
        g_pyramid = [np.repeat(g[:, :, np.newaxis], 3, axis=2) if g.ndim == 2 else g for g in g_pyramid]
        return g_pyramid

    l_pyramid = []
    for i in range(len(g_pyramid) - 1):
        size = (g_pyramid[i].shape[1], g_pyramid[i].shape[0])
        l_expanded = cv2.pyrUp(g_pyramid[i+1], dstsize=size)
        l_pyramid.append(cv2.subtract(g_pyramid[i], l_expanded))
    l_pyramid.append(g_pyramid[-1])  # Add the smallest level as is

    return l_pyramid

def blend_pyramids(l_pyramids, g_pyramids):
    blended_pyramid = []
    levels = len(l_pyramids[0])
    for i in range(levels):
        weight_sum = sum(g_pyramids[k][i] for k in range(len(g_pyramids)))
        weight_sum = np.clip(weight_sum, 1e-10, np.inf)
        blended_level = sum((g_pyramids[k][i] / weight_sum) * l_pyramids[k][i] for k in range(len(l_pyramids)))
        blended_pyramid.append(blended_level)
    return blended_pyramid

def collapse_pyramid(pyramid):
    image = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        image = cv2.pyrUp(image, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        image = cv2.add(image, pyramid[i])
    p2, p98 = np.percentile(image, (2, 98))
    image = np.clip((image - p2) / (p98 - p2), 0, 1)
    image = (image * 255).astype(np.uint8)
    return image

def exposure_fusion(directory, scene_name):
    images = load_images_from_directory(directory)
    weight_maps = [calculate_weight_map(img, 0.5, 2.0, 0.7) for img in images]
    max_level = 6

    l_pyramids = [build_pyramid(img, max_level, 'laplacian') for img in images]
    g_pyramids = [build_pyramid(w_map, max_level, 'gaussian') for w_map in weight_maps]

    blended_pyramid = blend_pyramids(l_pyramids, g_pyramids)
    result_image = collapse_pyramid(blended_pyramid)

    cv2.imwrite(f'Results/{scene_name}_fused_image.png', result_image)

    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Fused Image of {scene_name}')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    scene_name = 'Cave' 
    directory = f'Images/{scene_name}'  
    exposure_fusion(directory, scene_name)
