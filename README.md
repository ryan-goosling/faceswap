# FaceSwap

## Installation

Create env, install requirements

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Run
```bash
python src/infer.py \
  --image_path assets/image1.jpg \
  --mask_path assets/mask1.jpg \
  --seed 292 \
  --output_path my_result.jpg
```

## Architecture overview & Pipeline idea

1. **Data preprocessing**
![Image Preprocessed](assets/image_preprocessing.jpg)
   - Initially, the provided reference images did not all meet the criteria. One image was discarded due to strong glare on the eyes, bright lipstick, and an overall deviation from the desired style.
   - The remaining images were retouched to reduce the appearance of wrinkles, acne, and bags under the eyes.
   - All images were cropped to a 1024x1024 ratio to ensure consistency and compatibility with the model.

3. **Learning the LoRA model**  
   - The LoRA model was fine-tuned using carefully selected learning parameters, and the results were saved on HuggingFace.
   Check the trained model on HuggingFace: [Glam Person Initial LoRA](https://huggingface.co/biglebowski/glam_person_initial).
   - Fundamental parameters include a small learning rate (`2e-5`) for precise fine-tuning, a resolution of `1024` matching the preprocessed images, and a maximum of `800` training steps to prevent overfitting.
   - Additional parameters like gradient accumulation steps and gradient checkpointing were employed to stabilize training and efficiently manage memory.
   - Training command:
     ```
     !accelerate launch train_dreambooth_lora_sdxl.py \
       --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
       --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
       --instance_data_dir="glam_person_images/" \
       --output_dir="glam_person" \
       --caption_column="text" \
       --mixed_precision="fp16" \
       --instance_prompt="photo of a TOK person" \
       --resolution=1024 \
       --train_batch_size=1 \
       --gradient_accumulation_steps=4 \
       --gradient_checkpointing \
       --learning_rate=2e-5 \
       --snr_gamma=3.0 \
       --lr_scheduler="cosine_with_restarts" \
       --lr_warmup_steps=50 \
       --use_8bit_adam \
       --max_train_steps=800 \
       --checkpointing_steps=200 \
       --seed="42"
     ```

5. **Image processing**

   3.1 **Extracting the face mask of the input image**  
   - Although not implemented in the code, extracting a face mask is feasible using models like GroundingDINO combined with SAM.
   - Pre-prepared masks were generated using the resource: [Grounded SAM on Replicate](https://replicate.com/schananas/grounded_sam).

   3.2 **Preparation of canny edges for inpainting control**  
   - Canny edge detection is applied to generate control images for the inpainting process.
   - Additionally, the original image is cropped so that the face mask occupies roughly half of the image volume.       - The cropped section is resized to 1024x1024, aligning with the model’s training on profile photographs.

   3.3 **Using a pipeline with preset models**  
   - The pipeline utilizes a `StableDiffusionXLControlNetInpaintPipeline` that integrates a VAE, a ControlNet model, and LoRA weights.
   - This design leverages pretrained models to guide the inpainting process effectively.
   - A balanced set of parameters and prompts is crucial. The positive prompt "photo of a TOK person, natural, 8k" combined with a carefully crafted negative prompt (e.g., "blurry, low quality, low contrast, oversaturated colors, unrealistic plastic skin, gloss, plastic, distorted, bad anatomy, low resolution, noisy") helps filter out undesirable artifacts.
   - Inference parameters such as:
     - `controlnet_conditioning_scale`: 0.5
     - `guidance_scale`: 7.5
     - `num_inference_steps`: 45
     - `strength`: 1.0
     - `inpaint_strength`: 1.0
     - `denoising_strength`: 0.75  
     were chosen to strike a balance between detail preservation and smooth inpainting.

6. **Image post-processing**  
   - Although not implemented in the current code, post-processing steps could involve using various refiners to enhance facial aesthetics.
   - Preliminary tests with different refiners did not yield the desired results, indicating that further experimentation is needed for optimal enhancement.


## Example Results

| Source Image                    | Target Image                    | Output Image                    |
|---------------------------------|---------------------------------|---------------------------------|
| ![source](assets/image1.jpg)    | ![target](assets/target1.jpg)   | ![output](results/result1.jpg)  |
| ![source](assets/image2.jpg)    | ![target](assets/target2.jpg)   | ![output](results/result2.jpg)  |
| ![source](assets/image3.jpg)    | ![target](assets/target3.jpg)   | ![output](results/result3.jpg)  |

*(Replace the example images with your own to demonstrate the face swap process.)*
