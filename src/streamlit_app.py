import os
import subprocess
import numpy as np
import PIL
from PIL import Image, ImageFilter
import streamlit as st

from config import CONFIG
from pipeline import load_pipeline, run_inference
from image_processing import (paste_back,
                              detect_and_crop_face,
                              prepare_control_img)

# Cache the pipeline so it's loaded only once
@st.cache_resource
def get_pipeline():
    return load_pipeline(CONFIG)

def generate_image(image_path, prompt, negative_prompt, config_params, pipe):
    # Load the original image
    image = PIL.Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    bbox, mask = detect_and_crop_face(image_np[:, :, ::-1], scale_factor=2)
    cropped_face = image.crop(bbox)

    # Create necessary directories
    os.makedirs("tmp/images", exist_ok=True)
    os.makedirs("tmp/mask", exist_ok=True)

    # Save cropped face
    cropped_face.save("tmp/images/img.jpg")

    # Run the face parsing script
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

    # Prepare control image (Canny)
    control_pil = prepare_control_img(image)

    # Resize images
    cropped_mask = mask.resize((1024, 1024))
    cropped_control = control_pil.crop(bbox).resize((1024, 1024))
    cropped_image = cropped_face.resize((1024, 1024))

    CONFIG["num_inference_steps"] = config_params["num_inference_steps"]
    CONFIG["controlnet_conditioning_scale"] = config_params["controlnet_conditioning_scale"]
    CONFIG["guidance_scale"] = config_params["guidance_scale"]
    CONFIG["denoising_strength"] = config_params["denoising_strength"]
    CONFIG["inpaint_strength"] = config_params["inpaint_strength"]
    CONFIG["prompt"] = config_params["prompt"]
    CONFIG["negative_prompt"] = config_params["negative_prompt"]
    

    # Run inference using the cached pipeline
    inpainted_crop = run_inference(pipe, CONFIG, cropped_image, cropped_mask, cropped_control)

    # Paste the inpainted crop back onto the original image
    inpainted_image = paste_back(image, inpainted_crop, bbox)

    filename_params = (
        f"steps{config_params['num_inference_steps']}_"
        f"scale{config_params['controlnet_conditioning_scale']}_"
        f"guidance{config_params['guidance_scale']}_"
        f"denoise{config_params['denoising_strength']}_"
        f"inpaint{config_params['inpaint_strength']}_"
        f"prompt:{config_params['prompt']}"
    )
    filename = f"results/{str(image_path).split('/')[-1][:-4]}_{filename_params}.jpg"
    # Save the result
    inpainted_image.save(filename)
    print(f"Result saved at {filename}")
    return filename, inpainted_image

def main():
    st.set_page_config(layout="wide")  # Set page layout to wide mode

    # Load the pipeline once
    pipe = get_pipeline()

    # Layout: Sidebar for controls
    with st.sidebar:
        st.title("Settings")

        # Default values from CONFIG
        default_prompt = CONFIG.get("prompt", "A portrait of a woman with a smile")
        default_negative_prompt = CONFIG.get("negative_prompt", "No background")

        prompt = st.text_area("Prompt", default_prompt)
        negative_prompt = st.text_area("Negative Prompt", default_negative_prompt)

        # Hyperparameter sliders
        num_inference_steps = st.slider("Inference Steps", 1, 50, CONFIG.get("num_inference_steps", 20))
        controlnet_conditioning_scale = st.slider("ControlNet Scale", 0.1, 2.0, CONFIG.get("controlnet_conditioning_scale", 1.0))
        guidance_scale = st.slider("Guidance Scale", 0.0, 20.0, CONFIG.get("guidance_scale", 7.5))
        denoising_strength = st.slider("Denoising Strength", 0.0, 1.0, CONFIG.get("denoising_strength", 0.5))
        inpaint_strength = st.slider("Inpaint Strength", 0.0, 1.0, CONFIG.get("inpaint_strength", 0.75))

        config_params = {
            "num_inference_steps": num_inference_steps,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "guidance_scale": guidance_scale,
            "denoising_strength": denoising_strength,
            "inpaint_strength": inpaint_strength,
            "prompt": prompt,
            "negative_prompt": negative_prompt
        }

    # Page layout: Two columns (Original Image | Processed Image)
    col1, col2 = st.columns([1, 1])

   # with col1:
        #st.subheader("Original Image")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png"])

    if uploaded_image is not None:
        # Save uploaded image to a temp file
        image_path = f"tmp/{uploaded_image.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        with col1:
            st.image(image_path, caption="Uploaded Image")
        
        st.session_state["image_path"] = image_path

    # Button to start generation
    if st.button("Start Generation") and "image_path" in st.session_state:
        image_path = st.session_state["image_path"]
        result_path, result_image = generate_image(image_path, prompt, negative_prompt, config_params, pipe)

        # Display in the right column (always in the same place)
        with col2:
            #st.subheader("Generated Image")
            st.image(result_path, caption="Generated Image")

        # Download button
        st.download_button(
            label="Download Generated Image",
            data=open(result_path, "rb").read(),
            file_name=os.path.basename(result_path),
            mime="image/jpeg"
        )

if __name__ == "__main__":
    main()
