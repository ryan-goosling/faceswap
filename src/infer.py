import argparse
import cv2
import numpy as np
import PIL

from config import CONFIG
from pipeline import load_pipeline, run_inference
from utils import prepare_color_canny, crop_to_75_percent_face, paste_back

def main():
    parser = argparse.ArgumentParser(description="SDXL Inpainting with ControlNet (Canny).")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the original RGB image.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image (L channel).")
    parser.add_argument("--seed", type=int, default=292, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, default="inpainted_result.jpg", help="Path to save the result.")
    parser.add_argument("--use_color_canny", action="store_true", help="Use the color-canny function from utils.py")

    args = parser.parse_args()

    # 1. Load the original image and mask
    image = PIL.Image.open(args.image_path).convert("RGB")
    mask = PIL.Image.open(args.mask_path).convert("L")

    # Example of optional cropping (uncomment if needed)
    # image = PIL.Image.fromarray(np.array(image)[100:1124, 100:1124])
    # mask = PIL.Image.fromarray(np.array(mask)[100:1124, 100:1124])

    # 2. Generate the "control_image" (Canny)
    if args.use_color_canny:
        # We use the function from utils.py (semi-colored effect)
        control_pil = prepare_color_canny(np.array(image)[:, :, ::-1])
    else:
        # Regular Canny (black & white)
        control_input = cv2.Canny(np.array(image), 150, 200)
        control_input = cv2.cvtColor(control_input, cv2.COLOR_GRAY2RGB)
        control_pil = PIL.Image.fromarray(control_input)

    # 3. Initialize the pipeline (all parameters come from config.py)
    pipe = load_pipeline(CONFIG)

    cropped_image, cropped_mask, cropped_control, crop_coords = crop_to_75_percent_face(image, mask, control_pil)
    # 4. Run inference
    inpainted_crop = run_inference(pipe, CONFIG, cropped_image, cropped_mask, cropped_control, seed=args.seed)

    inpainted_image = paste_back(image, inpainted_crop, crop_coords)

    filename_params = (
        f"steps{CONFIG['num_inference_steps']}_"
        f"scale{CONFIG['controlnet_conditioning_scale']}_"
        f"guidance{CONFIG['guidance_scale']}_"
        f"denoise{CONFIG['denoising_strength']}"
    )
    filename = f"results/prompt_{str(args.image_path).split('/')[-1][:-4]}_{filename_params}.jpg"
    # 5. Save the result
    inpainted_image.save(filename)
    print(f"Result saved at {filename}")

if __name__ == "__main__":
    main()
