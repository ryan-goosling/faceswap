
import torch

CONFIG = {
    # Models
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "controlnet_id": "diffusers/controlnet-canny-sdxl-1.0",
    "vae_id": "madebyollin/sdxl-vae-fp16-fix",

    # а LoRA
    "lora_path": "biglebowski/glam_person_lora",
    #biglebowski/glam_person_lora

    # Prompts
    #"STRLK person, cinematic lighting, matte skin, ultra HD",
    #"prompt": "STRLK person, cinematic lighting, matte skin",
    "prompt": "STRLK person, cinematic lighting, matte skin, ultra HD", #matte skin,
    "negative_prompt": (
        "blurry, low quality, low contrast, oversaturated colors, unrealistic plastic skin, "
        "gloss, glare, plastic, distorted, wrinkles, low resolution, noisy"
    ),

    # Inference params
    "controlnet_conditioning_scale": 0.5,
    "guidance_scale": 7.5,
    "num_inference_steps": 30,
    "strength": 1.0,
    "inpaint_strength": 0.9,
    "denoising_strength": 0.5,

    # Tech
    "use_safetensors": True,
    "torch_dtype": torch.float16,
    "device": "cuda",
}
