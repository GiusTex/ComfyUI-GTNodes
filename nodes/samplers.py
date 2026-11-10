import torch

from comfy.cli_args import args
from comfy_api.latest import io, ui
from nodes import ConditioningZeroOut

import comfy.utils
import latent_preview
from comfy.samplers import SAMPLER_NAMES, sampler_object

from comfy_extras.nodes_custom_sampler import (Noise_RandomNoise, Noise_EmptyNoise, 
                                               SamplerCustomAdvanced, Guider_Basic)

from .samplers_utils.utils import (_h3_decode_all, _save_preview_webp, warning,
                                   set_preview_method, scene_starts, tiling_settings,
                                   check_guider, process_sigmas, check_sigmas,
                                   video_vae_decode_latent, audio_vae_decode_latent)

from .samplers_utils.vhs_patch import check_eligibility_to_patch, patch_vhs
from .tile_sampler_modules import h3_tiles, preview


class SamplerCustomAdvancedEfficient(io.ComfyNode):
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
            category="ComfyUI-GTNodes/samplers",
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
                io.Combo.Input("preview_method", options=["auto", "latent2rgb", "taesd", "taeh3", "vae_decoded_only", "none"], default="latent2rgb", 
                    tooltip=("\"taeh3\" requires \"taeh3.safetensors\" in models/vae_approx, "
                    "from kijai, and sets \"max_preview_resolution\" at 1024.\n"
                    "The animated preview requires https://github.com/kosinkadink/ComfyUI-VideoHelperSuite, "
                    "then: Settings -> 🎥🅥🅗🅢 -> \"Display animated previews when sampling\" ON. "
                    "You can also change video preview's fps here. For a single image VHS is not required.\n"
                    "\"taeh3\" previews have been tested both with single image and video generation. Decode tests (on a rtx 3060 12gb):\n"
                    "- 8 seconds, 448x704: the sampler stopped each step for 1.3 seconds;\n"
                    "- 8 seconds, 576x832: the sampler stopped each step for 2.1 seconds")),
                io.Combo.Input("vae_decode", options=["true", "true (tiled)", "false"], tooltip="Automatically decodes the denoised image/video latent.\n-true: forces image/video decoding;\n-true (tiled): forces image/video decoding using tiles\n-false: disables video decoding."),
                io.Combo.Input("audio_decode", options=["true", "true (tiled)", "false"], tooltip="Automatically decodes the denoised audio latent.\n-true: forces audio decoding;\n-true (tiled): forces audio decoding using tiles\n-false: disables audio decoding"),
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
    def define_preview(cls, has_images, preview_method, latent_to_use, images):
        ui_obj = None
        if not has_images and preview_method == "taeh3":
            samples = latent_to_use["samples"]
            if samples.is_nested:
                samples = samples.tensors[0]
            frames = _h3_decode_all(samples)
            entry = _save_preview_webp(frames, fps=8)
            ui_dict = {"images": [entry]}
            if frames.shape[0] > 1:
                ui_dict["animated"] = (True,)
            ui_obj = ui_dict
        elif has_images:
            ui_obj = ui.PreviewImage(images, cls=cls)

        return ui_obj

    @classmethod
    def execute(cls, add_noise, noise_seed, sampler, use_cfg, sigmas, latent, start_at_step, end_at_step, preview_method, vae_decode, audio_decode, model=(None), cfg=None, positive=None, custom_guider=None, custom_sampler=None, video_vae=(None), audio_vae=(None)):
        # If vae is not connected, disable vae decoding
        if video_vae == (None,) and vae_decode != "false":
            print(f"{warning('Sampler Custom Advanced Efficient Warning:')} No video_vae input detected, proceeding as if vae_decode was false.\n")
            vae_decode = "false"
        if audio_vae == (None,) and audio_decode != "false":
            print(f"{warning('Sampler Custom Advanced Efficient Warning:')} No audio_vae input detected, proceeding as if audio_decode was false.\n")
            audio_decode = "false"
        
        noise = Noise_RandomNoise(noise_seed) if add_noise else Noise_EmptyNoise()
        original_sigmas, sigmas = process_sigmas(sigmas, end_at_step, start_at_step)
        negative = use_cfg.get("negative", None)
        cfg = use_cfg.get("cfg", None)

        if positive is None and negative is None and custom_guider is None:
            raise ValueError("No guider input detected, a positive and negative must be provided")
        
        if positive is not None and negative is None:
            negative = ConditioningZeroOut().zero_out(positive)[0]

        guider = check_guider(model, custom_guider, cfg, positive, negative)
        sampler = sampler_object(sampler) if custom_sampler is None else custom_sampler
        # -----------------------------------------------------------------------

        try:
            # Initialize output variables
            images = audio = previous_preview_method = None
            previous_preview_method = args.preview_method
            previous_previewer = latent_preview.get_previewer

            latent_image = latent["samples"]
            if latent_image.is_nested:
                latent_image = latent_image.unbind()[0]
            patched, wrapped_previewer = check_eligibility_to_patch(latent_image, preview_method)
            set_preview_method(preview_method)

            (latent, denoised_latent) = SamplerCustomAdvanced().execute(
                noise=noise, guider=guider, sampler=sampler,
                sigmas=sigmas, latent_image=latent)

            latent_to_use = check_sigmas(end_at_step, original_sigmas, denoised_latent, latent)
            # -----------------------------------------------------------------------

            # Decode image if not yet decoded
            if images is None and "true" in vae_decode:
                images = video_vae_decode_latent(video_vae, latent_to_use, vae_decode)
            # Define a dummy output image
            if images is None and vae_decode == "false":
                images = list()
            
            has_images = images is not None and len(images) > 0
            ui_obj = cls.define_preview(has_images, preview_method, latent_to_use, images)

            # Decode audio if not yet decoded
            if "true" in audio_decode and latent_to_use["samples"].is_nested and audio is None:
                audio = audio_vae_decode_latent(audio_vae, latent_to_use, audio_decode)
            # Define a dummy output audio
            if audio is None and audio_decode == "false":
                audio = list()

        finally: # catch errors, exceptions etc
            # Restore global changes
            if patched:
                patch_vhs(False, wrapped_previewer)
            latent_preview.get_previewer = previous_previewer
            set_preview_method(previous_preview_method)
            torch.cuda.empty_cache()

        return io.NodeOutput(model, positive, negative, sampler, original_sigmas, latent, denoised_latent, images, audio, video_vae, audio_vae,
                            ui=ui_obj)


# Based on was-node-suite H3TiledSampler
class H3TiledSamplerCustomAdvancedEfficient(io.ComfyNode):
    """Denoise a MiniMax H3 latent as KSampler does, every model call run over overlapping tiles."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3TiledSamplerCustomAdvancedEfficient",
            display_name="H3 Tiled Sampler Custom Advanced (Efficient)",
            category="ComfyUI-GTNodes/samplers",
            description=(
                "Refine a long or upscaled MiniMax H3 clip with KSampler's settings, in overlapping tiles across the frame and windows along the clip that each fit the card, blended at every step into one clip. "
                "It does no upscaling itself. Each time window stays inside one scene, and the panel shows where the tiles sit. "
                "Requires a dict of video (or single image) + audio, and returns a dict."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 model, after any LoRA."),
                io.Sigmas.Input("sigmas"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True, tooltip="The noise seed, `0` as good as any."),
                io.Combo.Input("sampler", options=SAMPLER_NAMES),
                io.Sampler.Input("custom_sampler", optional=True),
                io.Conditioning.Input("conditioning", tooltip="The clip's positive conditioning."),
                io.Latent.Input("latent", tooltip="The H3 dict with upscaled vimage latent to refine + audio"),
                io.Vae.Input("video_vae", optional=True, tooltip="Required only for previews with \"auto\" option and vae_decode"),
                io.Vae.Input("audio_vae", optional=True, tooltip="Required only for vae_decode with models generating audio"),
                io.Int.Input("start_at_step", default=0, min=0, max=10000),
                io.Int.Input("end_at_step", default=10000, min=0, max=10000),
                io.Int.Input("tile_width", default=576, min=64, max=16384, step=32, tooltip="Pixels a tile spans across the frame, as `576`."),
                io.Int.Input("tile_height", default=832, min=64, max=16384, step=32, tooltip="Pixels a tile spans down the frame, as `832`."),
                io.Int.Input("overlap", default=128, min=32, max=4096, step=32, tooltip="Pixels neighbouring tiles share, as `128`."),
                io.Int.Input("window_frames", default=0, min=0, max=10000, step=1, tooltip="Frames a time window spans, as `141`, or `0` for the whole clip or a single image.",),
                io.Int.Input("window_overlap", default=17, min=1, max=1000, step=1, tooltip="Frames neighbouring time windows share, as `17` for a video or `1` for a single image."),
                io.Combo.Input("preview_method", ["auto", "latent2rgb", "taeh3", "none"], default="taeh3", 
                    tooltip=("\"taeh3\" requires \"taeh3.safetensors\" in models/vae_approx, "
                    "from kijai, and sets \"max_preview_resolution\" at 1024.\n"
                    "The animated preview requires https://github.com/kosinkadink/ComfyUI-VideoHelperSuite, "
                    "then: Settings -> 🎥🅥🅗🅢 -> \"Display animated previews when sampling\" ON. "
                    "You can also change video preview's fps here. For a single image VHS is not required.\n"
                    "\"taeh3\" previews have been tested both with single image and video generation. Decode tests (on a rtx 3060 12gb):\n"
                    "- 8 seconds, 448x704: the sampler stopped each step for 1.3 seconds;\n"
                    "- 8 seconds, 576x832: the sampler stopped each step for 2.1 seconds")),
                io.Combo.Input("vae_decode", options=["true", "true (tiled)", "false"], tooltip="Automatically decodes the denoised image/video latent.\n-true: forces image/video decoding;\n-true (tiled): forces image/video decoding using tiles\n-false: disables video decoding."),
                io.Combo.Input("audio_decode", options=["true", "true (tiled)", "false"], tooltip="Automatically decodes the denoised audio latent.\n-true: forces audio decoding;\n-true (tiled): forces audio decoding using tiles\n-false: disables audio decoding"),
            ],
            outputs=[
                io.Model.Output(display_name="model"), 
                io.Conditioning.Output(display_name="conditioning"),
                io.Sampler.Output(display_name="sampler"),
                io.Sigmas.Output(display_name="original sigmas"),
                io.Latent.Output("latent", tooltip="The H3 dict with refined latent image + audio"),
                io.Latent.Output(display_name="denoised latent", tooltip="The denoised H3 dict (if the sampling was stopped before the last sigma) with refined latent image + audio"),
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Vae.Output(display_name="video_vae"),
                io.Vae.Output(display_name="audio_vae"),
            ],
        )
    
    @classmethod
    def execute(cls, model, sigmas, seed, sampler, conditioning, latent, start_at_step, end_at_step, tile_width, tile_height, overlap, window_frames, window_overlap, preview_method, vae_decode, audio_decode, video_vae=(None), audio_vae=(None), custom_sampler=None):
        latent_dict = latent["samples"]
        if model.model.diffusion_model.__class__.__name__ == "MiniMaxH3ImageModel":
            latent_image = latent_dict
        else:
            if latent_dict.is_nested:
                latent_image = latent["samples"].unbind()[0]
            else:
                raise ValueError("[H3 Tiled Sampler] MiniMaxH3/P-H3-Image model detected, but the latent is not nested, use \"Concat AV Latent\" node to pass a video/image + audio nested latent. Or use MiniMaxH3-Image")

        # If vae is not connected, disable vae decoding
        if video_vae == (None,) and vae_decode != "false":
            print(f"{warning('Sampler Custom Advanced Efficient Warning:')} No video_vae input detected, proceeding as if vae_decode was false.\n")
            vae_decode = "false"
        if audio_vae == (None,) and audio_decode != "false":
            print(f"{warning('Sampler Custom Advanced Efficient Warning:')} No audio_vae input detected, proceeding as if audio_decode was false.\n")
            audio_decode = "false"
        
        starts = scene_starts(latent_image)
        settings = tiling_settings(tile_width, tile_height, overlap, window_frames, window_overlap)
        tiled_patcher = h3_tiles.tiled_model(model, settings, starts)
        
        samples = comfy.sample.fix_empty_latent_channels(
            tiled_patcher, latent_dict,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index"))

        original_sigmas, sigmas = process_sigmas(sigmas, end_at_step, start_at_step)

        guider=Guider_Basic(model)
        guider.set_conds(conditioning)
        sampler = sampler_object(sampler) if custom_sampler is None else custom_sampler

        try:
            # Initialize output variables
            images = audio = previous_preview_method = None
            previous_preview_method = args.preview_method
            previous_previewer = latent_preview.get_previewer

            patched, wrapped_previewer = check_eligibility_to_patch(latent_image, preview_method)
            set_preview_method(preview_method)

            x0_output = {}
            callback = preview.prepare_callback(tiled_patcher, len(sigmas) - 1, x0_output)

            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            samples = guider.sample(
                noise, samples, sampler, sigmas, 
                denoise_mask=latent.get("noise_mask", None), 
                callback=callback, disable_pbar=disable_pbar, 
                seed=seed)
            samples = samples.to(comfy.model_management.intermediate_device())

            out = latent.copy()
            out.pop("downscale_ratio_spacial", None)
            out.pop("downscale_ratio_temporal", None)
            out["samples"] = samples
            if "x0" in x0_output:
                x0 = x0_output["x0"]
                if samples.is_nested and not x0.is_nested:
                    latent_shapes = [x.shape for x in samples.unbind()]
                    x0 = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0, latent_shapes))
                x0_out = guider.model_patcher.model.process_latent_out(x0.cpu())
                out_denoised = latent.copy()
                out_denoised["samples"] = x0_out
            else:
                out_denoised = out

            latent_to_use = check_sigmas(end_at_step, original_sigmas, out_denoised, out)
            # -----------------------------------------------------------------------
                        
            # Decode image if not yet decoded
            if images is None and "true" in vae_decode:
                images = video_vae_decode_latent(video_vae, latent_to_use, vae_decode)
            # Define a dummy output image
            if images is None and vae_decode == "false":
                images = list()

            has_images = images is not None and len(images) > 0
            ui_obj = SamplerCustomAdvancedEfficient.define_preview(has_images, preview_method, latent_to_use, images)
            
            # Decode audio if not yet decoded
            if "true" in audio_decode and latent_to_use["samples"].is_nested and audio is None:
                audio = audio_vae_decode_latent(audio_vae, latent_to_use, audio_decode)
            # Define a dummy output audio
            if audio is None and audio_decode == "false":
                audio = list()

        finally: # catch errors, exceptions etc
            # Restore global changes
            if patched:
                patch_vhs(False, wrapped_previewer)
            latent_preview.get_previewer = previous_previewer
            set_preview_method(previous_preview_method)
            torch.cuda.empty_cache()

        return io.NodeOutput(model, conditioning, sampler, original_sigmas, out, out_denoised, images, audio, video_vae, audio_vae,
                            ui=ui_obj)


SAMPLERS_CLASS_MAPPINGS = {
    "SamplerCustomAdvancedEfficient": SamplerCustomAdvancedEfficient,
    "H3TiledSamplerCustomAdvancedEfficient": H3TiledSamplerCustomAdvancedEfficient,
}

SAMPLERS_NAME_MAPPINGS = {
    "SamplerCustomAdvancedEfficient": "Sampler Custom Advanced (Efficient)",
    "H3TiledSamplerCustomAdvancedEfficient": "H3 Tiled Sampler Custom Advanced (Efficient)",
}
