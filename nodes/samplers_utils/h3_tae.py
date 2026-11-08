import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file

import folder_paths
import latent_preview
import comfy.model_management as mm
from comfy.taesd.taesd import Block, Clamp


_CACHE = {}
MAX_RESOLUTION = 1024
PREVIEW_CHUNK = 8

# Taken from: https://github.com/jlucasmcrell/ComfyUI-H3-Multishot/blob/d7d197709fc93c1afb125e3db1dfd4ac0ae28c60/h3_tae_decode.py#L20
def _build_decoder(sd):
    idxs = sorted({int(k.split(".")[0]) for k in sd})
    mods, last_was_stem = [], False
    for i in range(idxs[-1] + 1):
        wk, bk = "%d.weight" % i, "%d.conv.0.weight" % i
        if wk in sd and sd[wk].ndim == 4:
            o, c, kh, kw = sd[wk].shape
            mods.append(nn.Conv2d(c, o, (kh, kw), padding=(kh // 2, kw // 2),
                                  bias=("%d.bias" % i) in sd))
            last_was_stem = (i == idxs[0])
        elif bk in sd:
            n_in = sd[bk].shape[1]
            n_out = sd["%d.conv.4.weight" % i].shape[0]
            mods.append(Block(n_in, n_out))
            last_was_stem = False
        else:
            if i == 0:
                mods.append(Clamp())
            elif last_was_stem:
                mods.append(nn.ReLU())
                last_was_stem = False
            else:
                mods.append(nn.Upsample(scale_factor=2, mode="nearest"))
    return nn.Sequential(*mods)


def _h3_get_net(tae_name="taeh3.safetensors"):
    path = folder_paths.get_full_path("vae_approx", tae_name)
    dev = mm.get_torch_device()
    dtype = mm.vae_dtype(dev, [torch.float16, torch.bfloat16])
    key = (path, str(dev), str(dtype))
    if key not in _CACHE:
        sd = load_file(path)
        net = _build_decoder(sd)
        net.load_state_dict(sd, strict=True)
        _CACHE[key] = net.to(dev, dtype).eval()
    return _CACHE[key], dev, dtype


class _TaeH3Adapter:
    # For VHS WrappedPreviewer: input [N,C,H,W], output [N,3,H,W] in -1..1
    def decode(self, x):
        net, dev, dtype = _h3_get_net()
        x = x.to(dev, dtype)
        outs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], PREVIEW_CHUNK):
                o = net(x[i:i + PREVIEW_CHUNK])
                outs.append(o.float().clamp(0, 1) * 2.0 - 1.0)
        out = torch.cat(outs, dim=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out

class H3TaePreviewer(latent_preview.LatentPreviewer):
    def __init__(self):
        self.taesd = _TaeH3Adapter() # now VHS can see it

    # H3 Tae Previewer (unpatched), called to generate previews 
    # when there is a single image or VHS is not installed/active
    def decode_latent_to_preview(self, x0):
        net, dev, dtype = _h3_get_net()
        lat = x0[0].movedim(1, 0)[:1] if x0.ndim == 5 else x0[:1]
        with torch.no_grad():
            out = net(lat.to(dev, dtype))[0]
            arr = (out.float().clamp(0, 1) * 255).to(torch.uint8)
        arr = arr.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(arr)
        return img

    def decode_latent_to_preview_image(self, preview_format, x0):
        # Server preview's structure: format, img, size
        return ("JPEG", self.decode_latent_to_preview(x0), MAX_RESOLUTION)
