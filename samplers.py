import torch
import json

from comfy.cli_args import args
from comfy_api.latest import io, IO
from nodes import VAEDecodeTiled, PreviewImage, VAEDecode, ConditioningZeroOut, EmptyImage

from comfy.samplers import SAMPLER_NAMES, CFGGuider, sampler_object

from comfy_extras.nodes_custom_sampler import (Noise_RandomNoise, Noise_EmptyNoise, 
                                               SamplerCustomAdvanced, Guider_Basic)

from comfy_extras.nodes_audio import VAEDecodeAudio, VAEDecodeAudioTiled, EmptyAudio

from .guider import ScheduledCfgGuider, CFGFloatListScheduler

from .utils import (global_preview_method, warning, 
                   set_preview_method, store_ksampler_results, globals_cleanup)


class SamplerCustomAdvancedEfficient(io.ComfyNode):
    # Image Preview code taken from jags111's efficiency-nodes (TSC_KSampler)
    empty_image = EmptyImage().generate(512, 512)[0]
    empty_audio = EmptyAudio.execute(duration=60.0, sample_rate=44100, channels=2)

    @classmethod
    def define_schema(cls):
        use_cfg_opt =[
            io.DynamicCombo.Option(
                "true", [
                    io.Conditioning.Input("negative", optional=True, tooltip="Not required if a guider is provided"),
                    io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01)
                ],
            ),
            io.DynamicCombo.Option(
                "false", [        
                ],
            ),
        ]

        return io.Schema(
            node_id="SamplerCustomAdvancedEfficient",
            display_name="Sampler Custom Advanced (Efficient)",
            description="Advanced ksampler that combines together some nodes, uses DynamicCombo which requires ComfyUI 0.8.1 and frontend 1.33.4 or later",
            category="More Efficient Samplers",
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent"),
                io.Model.Input("model", optional=True, tooltip="Required only if no guider is provided"),
                io.Conditioning.Input("positive", optional=True, tooltip="Not required if a guider is provided"),   
                io.Guider.Input("custom_guider", optional=True, tooltip="Custom guider to use instead of the default comfyui's CFG Guider. If no guider is connected, a positive and negative must be provided"),
                io.Sampler.Input("custom_sampler", optional=True, tooltip="Custom sampler to use instead of the default comfyui samplers"),
                io.Vae.Input("video_vae", optional=True, tooltip="Required only for previews with \"auto\" option and vae_decode"),
                io.Vae.Input("audio_vae", optional=True, tooltip="Required only for vae_decode with models generating audio"),
                io.Boolean.Input("add_noise", default=True),
                io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Combo.Input("sampler", options=SAMPLER_NAMES),
                io.DynamicCombo.Input("use_cfg", tooltip="Using cfg will use CFGGuider, and ConditioningZeroOut will be passed as negative if no negative is provided.\nDisabling cfg will use BasicGuider, which requires only the positive",
                    options=use_cfg_opt,
                ),
                io.Int.Input("start_at_step", default=0, min=0, max=10000),
                io.Int.Input("end_at_step", default=10000, min=0, max=10000),
                io.Combo.Input("preview_method", options=["auto", "latent2rgb", "taesd", "vae_decoded_only", "none"]),
                io.Combo.Input("vae_decode", options=["true", "true (tiled)", "false"], tooltip="Automatically decodes the denoised image/video latent.\n-true: forces image/video decoding;\n-true (tiled): forces image/video decoding using tiles\n-false: disables video decoding."),
                io.Combo.Input("audio_decode", options=["true", "true (tiled)", "false"], tooltip="Automatically decodes the denoised audio latent.\n-true: forces audio decoding;\n-true (tiled): forces audio decoding using tiles\n-false: disables audio decoding"),
            ],
            hidden=[
                IO.Hidden.prompt, 
                IO.Hidden.extra_pnginfo,
                IO.Hidden.unique_id
            ],
            outputs=[
                io.Model.Output(display_name="model"), 
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Sampler.Output(display_name="sampler"),
                io.Sigmas.Output(display_name="original sigmas"),
                io.Latent.Output(display_name="latent"),
                io.Latent.Output(display_name="denoised latent"),
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Vae.Output(display_name="video_vae"),
                io.Vae.Output(display_name="audio_vae"),
            ],
        )

    @classmethod
    def video_vae_decode_latent(cls, video_vae, out, vae_decode):
        if "tiled" in vae_decode:
            return VAEDecodeTiled().decode(video_vae, out, 320)[0]
        else:
            return VAEDecode().decode(video_vae, out)[0]

    @classmethod
    def audio_vae_decode_latent(cls, audio_vae, out, vae_decode):
        if "tiled" in vae_decode:
            return VAEDecodeAudioTiled().execute(audio_vae, out, 320, 64)[0]
        else:
            return VAEDecodeAudio().execute(audio_vae, out)[0]
    
    @classmethod
    def execute(cls, add_noise, noise_seed, sampler, use_cfg, sigmas, latent, start_at_step, end_at_step, preview_method, vae_decode, audio_decode, model=(None), cfg=None, positive=None, custom_guider=None, custom_sampler=None, video_vae=(None), audio_vae=(None)) -> io.NodeOutput:
        # Hidden args
        prompt_info = {}
        if cls.hidden.prompt is not None:
            prompt_info = cls.hidden.prompt

        metadata = {}
        if not args.disable_metadata:
            metadata["format"] = "pt"
            metadata["prompt"] = prompt_info
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])
        #--------------------------------------------------------------

        # If vae is not connected, disable vae decoding
        if video_vae == (None,) and vae_decode != "false":
            print(f"{warning('Sampler Custom Advanced Efficient Warning:')} No video_vae input detected, proceeding as if vae_decode was false.\n")
            vae_decode = "false"
        if audio_vae == (None,) and audio_decode != "false":
            print(f"{warning('Sampler Custom Advanced Efficient Warning:')} No audio_vae input detected, proceeding as if audio_decode was false.\n")
            audio_decode = "false"
        
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

        #-------------------------------------------------------------------
        negative = use_cfg.get("negative", None)
        cfg = use_cfg.get("cfg", None)
        #-------------------------------------------------------------------

        if positive is None and negative is None and custom_guider is None:
            raise ValueError("No guider input detected, a positive and negative must be provided")
        
        if positive is not None and negative is None:
            conditioningZeroOutNode = ConditioningZeroOut()
            negative, = conditioningZeroOutNode.zero_out(positive)

        #if custom_guider is None:
        if model is not None and custom_guider is None and cfg is None:
            guider = Guider_Basic(model)
            guider.set_conds(positive)
        elif model is not None and custom_guider is None and cfg is not None:
            guider = CFGGuider(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(cfg)
        elif custom_guider is not None:
            guider = custom_guider
        elif model is None and custom_guider is None:
            raise ValueError("Model is required when a guider is not provided!")
        
        if custom_sampler is not None:
            sampler = custom_sampler
        else:
            sampler = sampler_object(sampler)
        
        def process_latents(latent):
            # Initialize output variables
            images = audio = preview = previous_preview_method = None
        
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
                # Decode image/audio if not yet decoded
                if "true" in vae_decode:
                    if images is None:
                        images = cls.video_vae_decode_latent(video_vae, denoised_latent, vae_decode)
                        # Store decoded image as base image if no script is detected
                        store_ksampler_results("image", cls.hidden.unique_id, images)
                if "true" in audio_decode and denoised_latent["samples"].is_nested:
                    if audio is None:
                        audio = cls.audio_vae_decode_latent(audio_vae, denoised_latent, audio_decode)
                        # Store decoded audio as base audio if no script is detected
                        store_ksampler_results("audio", cls.hidden.unique_id, audio)

                # Define preview images
                if preview_method == "none" or (preview_method == "vae_decoded_only" and vae_decode == "false"):
                    preview = {"images": list()}
                elif images is not None:
                    preview = PreviewImage().save_images(images, prompt=prompt_info, extra_pnginfo=metadata)["ui"]

                # Define a dummy output image/audio
                if images is None and vae_decode == "false":
                    images = SamplerCustomAdvancedEfficient.empty_image
                if audio is None and audio_decode == "false":
                    audio = SamplerCustomAdvancedEfficient.empty_audio

            finally:
                # Restore global changes
                set_preview_method(previous_preview_method)
              
            return latent, denoised_latent, preview, images, audio
        
        # ---------------------------------------------------------------------------------------------------------------
        # Clean globally stored objects of non-existant nodes
        globals_cleanup(prompt_info)
        # ---------------------------------------------------------------------------------------------------------------
        latent, denoised_latent, preview, images, audio = process_latents(latent)

        result = io.NodeOutput(model, positive, negative, sampler, original_sigmas, latent, denoised_latent, images, audio, video_vae, audio_vae)

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
