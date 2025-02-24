import cv2
import numpy as np
from PIL import Image

def prepare_color_canny(image, low_threshold=50, high_threshold=200):
    """
    Processes the input image by applying Canny Edge Detection
    and blending the result over the original image.
    
    :param image: a NumPy array in BGR format (usually from cv2.imread)
    :param low_threshold: lower threshold for Canny
    :param high_threshold: upper threshold for Canny
    :return: a PIL.Image with the edges blended in
    """
    # Convert to RGB (for compatibility with diffusers)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale for the Canny algorithm
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract edges
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Convert edges to a 3-channel format
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Blend the original image with the edges (you can adjust the blend ratios)
    color_canny = cv2.addWeighted(image_rgb, 0.1, edges_colored, 0.9, 0)

    return Image.fromarray(color_canny)

def crop_to_75_percent_face(image, mask, control):
    """
    Crops the image so that 75% of the area is the face, based on the mask.
    """
    mask_np = np.array(mask)
    image_np = np.array(image)

    # Find bounding box of the face from the mask
    y, x = np.where(mask_np > 0)
    if len(x) == 0 or len(y) == 0:
        return image, mask  # No face detected, return original

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    face_w, face_h = x_max - x_min, y_max - y_min

    # Calculate target crop size so that 75% of area is the face
    target_area = (face_w * face_h) / 0.3
    target_size = int(np.sqrt(target_area))  # Square crop

    # Ensure the crop is centered on the face
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    crop_x_min = max(0, center_x - target_size // 2)
    crop_y_min = max(0, center_y - target_size // 2)
    crop_x_max = min(image_np.shape[1], center_x + target_size // 2)
    crop_y_max = min(image_np.shape[0], center_y + target_size // 2)

    # Crop and resize to 1024x1024 for SDXL
    cropped_image = image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max)).resize((1024, 1024))
    cropped_mask = mask.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max)).resize((1024, 1024))
    cropped_control = control.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max)).resize((1024, 1024))

    return cropped_image, cropped_mask, cropped_control, (crop_x_min, crop_y_min, crop_x_max, crop_y_max)

# Paste the result back into the original image
def paste_back(original_image, inpainted_crop, crop_coords):
    """Pastes the inpainted region back into the original image."""
    original_image = np.array(original_image)
    inpainted_crop = np.array(inpainted_crop.resize((crop_coords[2] - crop_coords[0], crop_coords[3] - crop_coords[1])))

    # Blend the images
    original_image[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]] = inpainted_crop

    return Image.fromarray(original_image)