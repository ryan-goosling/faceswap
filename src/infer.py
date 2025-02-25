import argparse
import cv2
import numpy as np
import PIL
from PIL import Image, ImageFilter
import os
import subprocess

from config import CONFIG
from pipeline import load_pipeline, run_inference
from image_processing import (paste_back,
                              detect_and_crop_face,
                              prepare_control_img)

def main():
    parser = argparse.ArgumentParser(description="SDXL Inpainting with ControlNet (Canny).")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the original RGB image.")

    args = parser.parse_args()

    # Load the original image and mask
    image = PIL.Image.open(args.image_path).convert("RGB")
    image_np = np.array(image)

    bbox, mask = detect_and_crop_face(image_np[:,:,::-1], scale_factor=2)
    cropped_face = image.crop(bbox)
    
    # Create necessary directories
    os.makedirs("tmp/images", exist_ok=True)
    os.makedirs("tmp/mask", exist_ok=True)

    # Load the original image and save it
    image = Image.open(args.image_path).convert("RGB")
    cropped_face.save("tmp/images/img.jpg")

    # Run the inference script
    subprocess.run([
        "python", "src/face_parser.py",
        "--model", "resnet18",
        "--weight", "face-parsing/weights/resnet18.pt",
        "--input", "tmp/images",
        "--output", "tmp/mask"
    ])

    # Load the binary mask
    mask = Image.open("tmp/mask/img.jpg").convert("L")
    mask = mask.filter(ImageFilter.MaxFilter(size=3))

    # Regular Canny (black & white)
    control_pil = prepare_control_img(image)

    # To same image size
    cropped_mask = mask.resize((1024, 1024))
    cropped_control = control_pil.crop((bbox)).resize((1024, 1024))
    cropped_image = cropped_face.resize((1024, 1024))

    # Initialize the pipeline (all parameters come from config.py)
    # Run inference
    pipe = load_pipeline(CONFIG)
    
    inpainted_crop = run_inference(pipe, CONFIG, cropped_image, cropped_mask, cropped_control)
    
    # Debug visualisation
    cropped_mask.save("mask.jpg")
    cropped_control.save("control.jpg")
    cropped_image.save("img.jpg")
    inpainted_crop.save("inpainted.jpg")

    # Get final result with source image
    inpainted_image = paste_back(image, inpainted_crop, bbox)

    filename_params = (
        f"steps{CONFIG['num_inference_steps']}_"
        f"scale{CONFIG['controlnet_conditioning_scale']}_"
        f"guidance{CONFIG['guidance_scale']}_"
        f"denoise{CONFIG['denoising_strength']}_"
        f"inpaint{CONFIG['inpaint_strength']}_"
        f"prompt:{CONFIG['prompt']}"
    )
    filename = f"results/{str(args.image_path).split('/')[-1][:-4]}_{filename_params}.jpg"
    # Save the result
    inpainted_image.save(filename)
    print(f"Result saved at {filename}")

if __name__ == "__main__":
    main()
