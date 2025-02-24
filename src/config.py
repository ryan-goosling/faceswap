
import torch

CONFIG = {
    # Models
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "controlnet_id": "diffusers/controlnet-canny-sdxl-1.0",
    "vae_id": "madebyollin/sdxl-vae-fp16-fix",

    # а LoRA
    "lora_path": "biglebowski/glam_person_initial",
    #"lora_path": "biglebowski/glam_person_initial",

    # Prompts
    "prompt": "photo of a TOK person, natural, 8k",
    "negative_prompt": (
        "blurry, low quality, low contrast, oversaturated colors, unrealistic plastic skin, "
        "gloss, plastic, distorted, bad anatomy, low resolution, noisy"
    ),

    # Inference params
    "controlnet_conditioning_scale": 0.5,
    "guidance_scale": 7.5,
    "num_inference_steps": 45,
    "strength": 1.0,
    "inpaint_strength": 1.0,
    "denoising_strength": 0.75,

    # Tech
    "use_safetensors": True,
    "torch_dtype": torch.float16,
    "device": "cuda",
}
