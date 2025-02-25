import sys
import os

# Dynamically get the absolute path of face-parsing and add it to sys.path
face_parsing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../face-parsing"))
if face_parsing_path not in sys.path:
    sys.path.insert(0, face_parsing_path)  # Prioritize face-parsing module

# Ensure correct imports
from models.bisenet import BiSeNet
import argparse
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms


def prepare_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image_tensor = transform(image)
    image_batch = image_tensor.unsqueeze(0)
    return image_batch

@torch.no_grad()
def inference(config):
    output_path = config.output
    input_path = config.input
    weight = config.weight
    model_name = config.model

    output_path = os.path.join(output_path)
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19

    model = BiSeNet(num_classes, backbone_name=model_name)
    model.to(device)

    if os.path.exists(weight):
        model.load_state_dict(torch.load(weight, map_location=device))
    else:
        raise ValueError(f"Weights not found from given path ({weight})")

    if os.path.isfile(input_path):
        input_path = [input_path]
    else:
        input_path = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.jpg', '.png'))]
    
    model.eval()
    for file_path in input_path:
        image = Image.open(file_path).convert("RGB")
        print(f"Processing image: {file_path}")

        resized_image = image.resize((512, 512), resample=Image.BILINEAR)
        transformed_image = prepare_image(resized_image)
        image_batch = transformed_image.to(device)

        output = model(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

        # Define class-to-color mapping
        # Define class-to-color mapping for up to 17 classes
        COLOR_MAP = {
            1: (255, 0, 0),    # Skin - Red
            2: (0, 255, 0),    # Left Eyebrow - Green
            3: (0, 0, 255),    # Right Eyebrow - Blue
            4: (255, 255, 0),  # Left Eye - Yellow
            5: (255, 0, 255),  # Right Eye - Magenta
            6: (0, 255, 255),  # Nose - Cyan
            7: (128, 0, 0),    # Upper Lip - Dark Red
            8: (0, 128, 0),    # Inner Mouth - Dark Green
            9: (0, 0, 128),    # Lower Lip - Dark Blue
            10: (128, 128, 0), # Hair - Olive
            11: (128, 0, 128), # Ear - Purple
            12: (0, 128, 128), # Neck - Teal
            13: (64, 0, 0),    # Cloth - Darker Red
            14: (0, 64, 0),    # Hat - Darker Green
            15: (0, 0, 64),    # Glasses - Darker Blue
            16: (192, 192, 192), # Accessories - Silver
            17: (255, 165, 0), # Background - Orange
        }
        
        # Create a blank RGB image for the mask
        colored_mask = np.zeros((*predicted_mask.shape, 3), dtype=np.uint8)

        # Define the list of class indexes to color
        selected_classes = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13]  # Change this list as needed
        
        # Apply colors only to selected classes
        for class_idx in selected_classes:
            if class_idx in COLOR_MAP:
                colored_mask[predicted_mask == class_idx] = COLOR_MAP[class_idx]
        
        # Convert to image and save
        binary_mask = np.isin(predicted_mask, [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13]).astype(np.uint8) * 255
        binary_mask_image = Image.fromarray(binary_mask)
        binary_mask_image.save(os.path.join(output_path, os.path.basename(file_path)))

def parse_args():
    parser = argparse.ArgumentParser(description="Face parsing inference")
    parser.add_argument("--model", type=str, default="resnet18", help="model name, i.e resnet18, resnet34")
    parser.add_argument("--weight", type=str, default="face_parsing/weights/resnet18.pt", help="path to trained model")
    parser.add_argument("--input", type=str, default="assets/images", help="path to an image or a folder of images")
    parser.add_argument("--output", type=str, default="assets/results", help="path to save model outputs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inference(config=args)
