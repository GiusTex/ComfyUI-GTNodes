import os
import random
import math
import torch
from PIL import Image

import folder_paths
import latent_preview
from comfy.cli_args import args
from comfy.samplers import CFGGuider
from nodes import VAEDecodeTiled, VAEDecode
from comfy_extras.nodes_custom_sampler import Guider_Basic
from comfy_extras.nodes_audio import VAEDecodeAudio, VAEDecodeAudioTiled

from .h3_tae import _h3_get_net


# -------------------- SamplerAdvanced utils --------------
WARNING_COLOR = "\033[93m" # Yellow

def format_message(text, color_code):
    RESET_COLOR = "\033[0m"
    return f"{color_code}{text}{RESET_COLOR}"

def warning(text):
    return format_message(text, WARNING_COLOR)

# Set global preview_method
def set_preview_method(method):
    if method == 'auto' or method == 'LatentPreviewMethod.Auto':
        args.preview_method = latent_preview.LatentPreviewMethod.Auto
    elif method == 'latent2rgb' or method == 'LatentPreviewMethod.Latent2RGB':
        args.preview_method = latent_preview.LatentPreviewMethod.Latent2RGB
    elif method == 'taesd' or method == 'LatentPreviewMethod.TAESD':
        args.preview_method = latent_preview.LatentPreviewMethod.TAESD
    else:
        args.preview_method = latent_preview.LatentPreviewMethod.NoPreviews

# Basic functions, some used by both samplers, dumped here to clear space in nodes.py
def check_guider(model, custom_guider, cfg, positive, negative):
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
    return guider

def process_sigmas(sigmas, end_at_step, start_at_step):
    original_sigmas = sigmas.clone()
    if end_at_step < (len(sigmas) - 1):
        sigmas = sigmas[:end_at_step + 1]
        
    if start_at_step > 0:
        if start_at_step < (len(sigmas) - 1):
            sigmas = sigmas[start_at_step:]
        else:
            raise ValueError("Start step is greater than the number of steps in sigmas, cannot proceed")
        
    return original_sigmas, sigmas

def check_sigmas(end_at_step, original_sigmas, denoised_latent, latent):
    if end_at_step < len(original_sigmas) - 1:
        return denoised_latent
    else:
        return latent

def video_vae_decode_latent(video_vae, out, vae_decode):
    if "tiled" in vae_decode:
        return VAEDecodeTiled().decode(video_vae, out, 320)[0]
    else:
        return VAEDecode().decode(video_vae, out)[0]

def audio_vae_decode_latent(audio_vae, out, vae_decode):
    if "tiled" in vae_decode:
        return VAEDecodeAudioTiled().execute(audio_vae, out, 320, 64)[0]
    else:
        return VAEDecodeAudio().execute(audio_vae, out)[0]

# --------- SamplerAdvanced Create Image/Video Preview ----------
# Accepts both video/single image
def _h3_decode_all(samples, chunk=8):
    # 5D [B,C,T,H,W] o 4D [B,C,H,W] -> uint8 [N,H,W,3] su CPU
    net, dev, dtype = _h3_get_net()
    lat = samples[0].movedim(1, 0) if samples.ndim == 5 else samples   # [N,C,H,W]
    out = []
    with torch.no_grad():
        for i in range(0, lat.shape[0], chunk):
            o = net(lat[i:i + chunk].to(dev, dtype))
            o = (o.float().clamp(0, 1) * 255).to(torch.uint8)
            out.append(o.permute(0, 2, 3, 1).cpu())
    return torch.cat(out, dim=0)

# Accepts both video/single image
def _save_preview_webp(frames_u8, fps=8, quality=80):
    pil = [Image.fromarray(f.numpy()) for f in frames_u8]
    name = f"h3_preview_{random.getrandbits(32):08x}.webp"
    path = os.path.join(folder_paths.get_temp_directory(), name)
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=int(1000 / fps), loop=0, quality=quality)
    return {"filename": name, "subfolder": "", "type": "temp"}


# -------------------- H3 Tiled Sampler ---------------------------
# Functions taken from was-node-suite h3_tiles.py
#: Latent key holding the token each fresh scene of a joined clip opens on, the first at 0.
SCENES_KEY = "h3_scene_starts"

#: Latent rows in one H3 chunk; a time window starts on a chunk.
CHUNK_ROWS = 5

#: Pixels per latent pixel in the H3 video latent.
LATENT_SCALE = 16

def scene_starts(latent: dict) -> list[int] | None:
    """The token each fresh scene of a joined clip opens on, as H3 Extend Append recorded it.

    Args:
        latent: A joined clip.

    Returns:
        Token numbers in order, the first ``0``, or ``None`` where nothing was recorded.
    """
    starts = latent.get(SCENES_KEY) if isinstance(latent, dict) else None
    return [int(start) for start in starts] if starts else None


def _rows(frames: int) -> int:
    """Latent rows covering ``frames`` video frames."""
    return max(1, math.ceil(int(frames) * CHUNK_ROWS / 17))

def tiling_settings(tile_width, tile_height, overlap, window_frames=0, window_overlap=1):
    def latent(pixels):
        return max(2, -(-int(pixels) // (2 * LATENT_SCALE)) * 2)

    return ("manual", latent(tile_height), latent(tile_width),
            _rows(window_frames) if window_frames > 0 else 0, 
            latent(overlap), _rows(window_overlap))
