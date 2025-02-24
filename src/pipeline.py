import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline, AutoencoderKL

def load_pipeline(cfg):
    """
    Creates and returns a StableDiffusionXLControlNetInpaintPipeline
    based on parameters from the cfg dictionary.
    """
    # If dtype is specified as a string (e.g., "torch.float16"), convert it:
    if isinstance(cfg["torch_dtype"], str):
        dtype = eval(cfg["torch_dtype"])
    else:
        dtype = cfg["torch_dtype"]

    # 1. Load the VAE
    vae = AutoencoderKL.from_pretrained(
        cfg["vae_id"],
        torch_dtype=dtype
    )

    # 2. Load ControlNet
    controlnet = ControlNetModel.from_pretrained(
        cfg["controlnet_id"],
        torch_dtype=dtype
    )

    # 3. Create the pipeline
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        cfg["model_id"],
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=cfg["use_safetensors"]
    )

    # 4. Load LoRA weights (replace with your own if necessary)
    pipe.load_lora_weights(cfg["lora_path"])

    # 5. Move the pipeline to the specified device (cuda / cpu / mps)
    pipe.to(cfg["device"])

    return pipe


def run_inference(pipe, cfg, image, mask, control_pil, seed=292):
    """
    Launches the inpainting process according to the parameters in cfg.
    Returns the generated image (PIL.Image).
    """
    #generator = torch.Generator(device=cfg["device"]).manual_seed(seed)

    result = pipe(
        prompt=cfg["prompt"],
        negative_prompt=cfg["negative_prompt"],
        image=image,
        mask_image=mask,
        control_image=control_pil,
        controlnet_conditioning_scale=cfg["controlnet_conditioning_scale"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        strength=cfg["strength"],
        inpaint_strength=cfg["inpaint_strength"],
        denoising_strength=cfg["denoising_strength"]
        #generator=generator
    )

    return result.images[0]
