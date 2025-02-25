import cv2
import numpy as np
import PIL
from PIL import Image


# Paste the result back into the original image
def paste_back(original_image, inpainted_crop, crop_coords):
    """Pastes the inpainted region back into the original image."""
    original_image = np.array(original_image)
    inpainted_crop = np.array(inpainted_crop.resize((crop_coords[2] - crop_coords[0], crop_coords[3] - crop_coords[1])))

    # Blend the images
    original_image[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]] = inpainted_crop

    return Image.fromarray(original_image)


def detect_and_crop_face(image, scale_factor=2, min_neighbors=8, min_size=(50, 50)):
    """
    Detects the largest face in an image, enlarges the bounding box, and returns
    the bounding box coordinates (x_min, y_min, x_max, y_max) and cropped image.

    :param image: Input image (NumPy array)
    :param scale_factor: Factor by which to enlarge the bounding box (default=1.3)
    :param min_neighbors: Minimum neighbors for Haar cascade (default=8)
    :param min_size: Minimum size for face detection (default=(50, 50))
    :return: (bbox, cropped_face) where bbox is (x_min, y_min, x_max, y_max) and cropped_face is the enlarged face region.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=min_size)

    if len(faces) > 0:
        # Select the largest face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        # Increase bbox size
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Center the new bbox
        x_min = max(0, x - (new_w - w) // 2)
        y_min = max(0, y - (new_h - h) // 2)

        # Ensure bbox stays within image bounds
        x_min = min(x_min, image.shape[1] - new_w)
        y_min = min(y_min, image.shape[0] - new_h)
        new_w = min(new_w, image.shape[1] - x_min)
        new_h = min(new_h, image.shape[0] - y_min)

        # Calculate the max coordinates
        x_max = x_min + new_w
        y_max = y_min + new_h

        # Crop the enlarged face region
        cropped_face = image[y_min:y_max, x_min:x_max]

        return (x_min, y_min, x_max, y_max), cropped_face
    
    return None, None

def prepare_control_img(image):
    control_input = cv2.Canny(np.array(image), 150, 200)
    control_input = cv2.cvtColor(control_input, cv2.COLOR_GRAY2RGB)
    control_pil = PIL.Image.fromarray(control_input)
    return control_pil
