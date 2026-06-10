import torch
from PIL import Image

from nodes import VAEDecodeTiled, PreviewImage, VAEDecode, ConditioningZeroOut
from comfy.samplers import SAMPLER_NAMES, CFGGuider, sampler_object
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, Noise_EmptyNoise, SamplerCustomAdvanced
from .guider import ScheduledCfgGuider, CFGFloatListScheduler

from .utils import (pil2tensor, global_preview_method, warning, 
                   set_preview_method, store_ksampler_results, globals_cleanup)


class SamplerCustomAdvancedEfficient:
    # Image Preview code taken from jags111's efficiency-nodes (TSC_KSampler)
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                    "sampler": (SAMPLER_NAMES,),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sigmas": ("SIGMAS",),
                    "latent": ("LATENT",),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "preview_method": (["auto", "latent2rgb", "taesd", "vae_decoded_only", "none"],),
                    "vae_decode": (["true", "true (tiled)", "false"], {"tooltip": "Automatically decode the denoised latent"}),
                    },
                    "optional": {
                        "positive": ("CONDITIONING", {"tooltip": "Not required if a guider is provided"}),
                        "negative": ("CONDITIONING", {"tooltip": "Not required if a guider is provided. If instead a positive but not a negative is provided, \"ConditioningZeroOut\" will be automatically used, and low cfg values would be suggested"}),
                        "custom_guider": ("GUIDER", {"tooltip": "Custom guider to use instead of the default comfyui's CFG Guider. If no guider is connected, a positive and negative must be provided"}),
                        "custom_sampler": ("SAMPLER", {"tooltip": "Custom sampler to use instead of the default comfyui samplers"}),
                        "vae": ("VAE", {"tooltip": "Required only for previews with \"auto\" option and vae_decode"}),
                    },
                    "hidden": {
                        "prompt": "PROMPT", 
                        "extra_pnginfo": "EXTRA_PNGINFO", 
                        "my_unique_id": "UNIQUE_ID",
                    },
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "SAMPLER", "SIGMAS", "LATENT", "LATENT", "IMAGE", "VAE",)
    RETURN_NAMES = ("model", "positive", "negative", "sampler", "original sigmas", "latent", "denoised latent", "images", "vae",)
    FUNCTION = "sample"
    CATEGORY = "More Efficient Samplers"

    def sample(cls, model, add_noise, noise_seed, sampler, cfg, sigmas, latent, start_at_step, end_at_step, preview_method, vae_decode, positive=None, negative=None, custom_guider=None, custom_sampler=None, vae=(None,), prompt=None, extra_pnginfo=None, my_unique_id=None):
        # If vae is not connected, disable vae decoding
        if vae == (None,) and vae_decode != "false":
            print(f"{warning('Sampler Custom Ultra Advanced Warning:')} No vae input detected, proceeding as if vae_decode was false.\n")
            vae_decode = "false"
        
        # ------------------------------------------------------------------------------------------------------
        def vae_decode_latent(vae, out, vae_decode):
            if "tiled" in vae_decode:
                return VAEDecodeTiled().decode(vae, out, 320)[0]
            else:
                return VAEDecode().decode(vae, out)[0]
        # ---------------------------------------------------------------------------------------------------------------
        
        if not add_noise:
            noise = Noise_EmptyNoise()
        else:
            noise = Noise_RandomNoise(noise_seed)
        
        original_sigmas = sigmas.clone()
        if end_at_step is not None and end_at_step < (len(sigmas) - 1):
            sigmas = sigmas[:end_at_step + 1]

        if start_at_step is not None:
            if start_at_step < (len(sigmas) - 1):
                sigmas = sigmas[start_at_step:]
            else:
                if latent["samples"] is not None:
                    return latent["samples"]
                else:
                    return torch.zeros_like(noise)
        
        if positive is None and negative is None and custom_guider is None:
            raise ValueError("No guider input detected, a positive and negative must be provided")
        
        if positive is not None and negative is None:
            conditioningZeroOutNode = ConditioningZeroOut()
            negative, = conditioningZeroOutNode.zero_out(positive)
        
        if custom_guider is None:
            guider = CFGGuider(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(cfg)
        else:
            guider = custom_guider
        
        if custom_sampler is not None:
            sampler = custom_sampler
        else:
            sampler = sampler_object(sampler)
        
        def process_latents(latent):
            # Initialize output variables
            images = preview = previous_preview_method = None
        
            try:
                # Change the global preview method (temporarily)
                set_preview_method(preview_method)
                
                # Using SamplerCustomAdvanced for NestedTensor and future comfyui compatibility and updates
                (latent, denoised_latent) = SamplerCustomAdvanced().execute(
                        noise=noise, guider=guider, sampler=sampler,
                        sigmas=sigmas, latent_image=latent
                )

                previous_preview_method = global_preview_method()

                # ---------------------------------------------------------------------------------------------------------------
                # Decode image if not yet decoded
                if "true" in vae_decode:
                    if images is None:
                        images = vae_decode_latent(vae, denoised_latent, vae_decode)
                        # Store decoded image as base image of no script is detected
                        store_ksampler_results("image", my_unique_id, images)

                # Define preview images
                if preview_method == "none" or (preview_method == "vae_decoded_only" and vae_decode == "false"):
                    preview = {"images": list()}
                elif images is not None:
                    preview = PreviewImage().save_images(images, prompt=prompt, extra_pnginfo=extra_pnginfo)["ui"]

                # Define a dummy output image
                if images is None and vae_decode == "false":
                    images = SamplerCustomAdvancedEfficient.empty_image

            finally:
                # Restore global changes
                set_preview_method(previous_preview_method)
              
            return latent, denoised_latent, preview, images
        
        # ---------------------------------------------------------------------------------------------------------------
        # Clean globally stored objects of non-existant nodes
        globals_cleanup(prompt)
        # ---------------------------------------------------------------------------------------------------------------
        latent, denoised_latent, preview, images = process_latents(latent)

        result = (model, positive, negative, sampler, original_sigmas, latent, denoised_latent, images, vae,)

        if preview is None:
            return {"result": result}
        else:
            return {"ui": preview, "result": result}


NODE_CLASS_MAPPINGS = {
    "SamplerCustomAdvancedEfficient": SamplerCustomAdvancedEfficient,
    "ScheduledCfgGuider": ScheduledCfgGuider,
    "CFGFloatListScheduler": CFGFloatListScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerCustomAdvancedEfficient": "Sampler Custom Advanced (Efficient)",
    "ScheduledCfgGuider": "Scheduled Cfg Guider",
    "CFGFloatListScheduler": "CFG Float List Scheduler",
}
