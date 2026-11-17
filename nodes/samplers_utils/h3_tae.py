import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file

import folder_paths
import latent_preview
from comfy.cli_args import args
import comfy.model_management as mm
from comfy.taesd.taesd import Block, Clamp

_CACHE = {}
PREVIEW_CHUNK = 8
# latent scale based on exit heads
SCALE = {5: 1, 10: 2, 15: 4}

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

# Process taeh3_image tiny vae's 3 exit heads
class _TaeH3Net(nn.Module):
    def __init__(self, sd):
        super().__init__()
        main = {k: v for k, v in sd.items() if k.split(".")[0].isdigit()}
        self.body = _build_decoder(main)
        self.body.load_state_dict(main, strict=True)

        self.exits = nn.ModuleDict()
        for k, w in sd.items():
            if k.startswith("exit.") and k.endswith(".weight"):
                n = k.split(".")[1]
                conv = nn.Conv2d(w.shape[1], w.shape[0], tuple(w.shape[2:]),
                                 padding=w.shape[2] // 2)
                conv.weight.data.copy_(w)
                conv.bias.data.copy_(sd[f"exit.{n}.bias"])
                self.exits[n] = conv

    # For MiniMax-H3, and MiniMax-H3-Image with exit_at (5, 10, 15)
    def forward(self, x, exit_at=None):
        if exit_at is None or str(exit_at) not in self.exits:
            return self.body(x) # No exit heads found: full decoder
        for m in self.body[:exit_at + 1]: # Modules 0..exit_at included
            x = m(x)
        return self.exits[str(exit_at)](x)

def _h3_get_net(tae_name):
    path = folder_paths.get_full_path("vae_approx", tae_name)
    dev = mm.get_torch_device()
    dtype = mm.vae_dtype(dev, [torch.float16, torch.bfloat16])
    key = (path, str(dev), str(dtype))
    if key not in _CACHE:
        sd = load_file(path)
        # filter keys and load body + exit heads
        net = _TaeH3Net(sd)
        _CACHE[key] = net.to(dev, dtype).eval()
    return _CACHE[key], dev, dtype

def _pick_exit(h, w):
    side = max(h, w)
    target = min(args.max_taeh3_preview_res, side * PREVIEW_CHUNK)
    for n in (5, 10, 15):
        if side * SCALE[n] >= target:
            return n
    return None # full decoder

class _TaeH3Adapter:
    def decode(self, x):
        net, dev, dtype = _h3_get_net("taeh3.safetensors")
        x = x.to(dev, dtype)
        outs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], PREVIEW_CHUNK):
                chunk = x[i:i + PREVIEW_CHUNK]
                o = net(chunk, exit_at=_pick_exit(chunk.shape[-2], chunk.shape[-1]))
                outs.append(o.float().clamp(0, 1) * 2.0 - 1.0)
        out = torch.cat(outs, dim=0)
        return out

class H3TaePreviewer(latent_preview.LatentPreviewer):
    def __init__(self):
        self.taesd = _TaeH3Adapter() # now VHS can see it

    # H3 Tae Previewer (unpatched), called to generate previews 
    # when there is a single image or VHS is not installed/active
    def decode_latent_to_preview(self, x0):
        net, dev, dtype = _h3_get_net("taeh3_image.safetensors")
        lat = x0[0].movedim(1, 0)[:1] if x0.ndim == 5 else x0[:1]
        with torch.no_grad():
            out = net(lat.to(dev, dtype),
                    exit_at=_pick_exit(lat.shape[-2], lat.shape[-1]))[0]
            arr = (out.float().clamp(0, 1) * 255).to(torch.uint8)
        arr = arr.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(arr)
        return img

    def decode_latent_to_preview_image(self, preview_format, x0):
        # Server preview's structure: format, img, size
        return ("JPEG", self.decode_latent_to_preview(x0), args.max_taeh3_preview_res)
