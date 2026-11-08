import torch
import torch.nn.functional as F
from PIL import Image

import latent_preview

import server
from io import BytesIO
import struct
import time

from ..samplers_utils.h3_tae import H3TaePreviewer, MAX_RESOLUTION


# --------------------- VHS Animated Preview Patch -----------------
_VHS_ORIG = {}
rate = 8 # as default in VHS's "Playback rate override"

try:
    previewers = list(latent_preview.LatentPreviewer.__subclasses__())
    for previewer in previewers:
        if previewer.__name__ == "WrappedPreviewer" and hasattr(previewer, "process_previews"):
            VHS_WRAPPER = previewer
except:
    None

# decode_latent_to_preview_image and process_previews copied from WrappedPreviewer in Kosinkadink/ComfyUI-VideoHelperSuite: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/videohelpersuite/latent_preview.py#L18
# and patched to accept H3TaePreviewer and MAX_RESOLUTION
def patch_vhs(on, wrapped_previewer):
    # Leaving the print here to show when the patch is turned on/off
    print(f"[ComfyUI-GTNodes] VHS's WrappedPreviewer TAEH3 patch: {"ON" if on else "OFF"}", flush=True)
    
    if not on: # Turn off at end of sampling or if there are errors, exceptions etc
        for k, v in _VHS_ORIG.items():
            setattr(wrapped_previewer, k, v)
        _VHS_ORIG.clear()
        return

    # Save originals if not already done (if patch remained active)
    if not getattr(wrapped_previewer.process_previews, "_h3_patched", False):
        _VHS_ORIG["decode_latent_to_preview_image"] = wrapped_previewer.decode_latent_to_preview_image
        _VHS_ORIG["process_previews"] = wrapped_previewer.process_previews

    serv = server.PromptServer.instance

    # VHS patched H3 Tae Previewer, called to generate previews 
    # of a video and when VHS is installed and active
    def decode_latent_to_preview_image(self, preview_format, x0):
        if x0.ndim == 5:
            x0 = x0.movedim(2, 1)
            x0 = x0.reshape((-1,) + x0.shape[-3:])
        num_images = x0.size(0)
        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews / self.rate
        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None
        if self.first_preview:
            self.first_preview = False
            serv.send_sync('VHS_latentpreview',
                           {'length': num_images, 'rate': self.rate, 'id': serv.last_node_id})
            self.last_time = new_time + 1 / self.rate
        if self.c_index + num_previews > num_images:
            x0 = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            x0 = x0[self.c_index:self.c_index + num_previews]
        self.process_previews(x0, self.c_index, num_images)
        self.c_index = (self.c_index + num_previews) % num_images
        return None

    def process_previews(self, image_tensor, ind, leng):
        image_tensor = self.decode_latent_to_preview(image_tensor) # [n,H,W,3], -1..1
        if image_tensor.size(1) > MAX_RESOLUTION or image_tensor.size(2) > MAX_RESOLUTION:
            image_tensor = image_tensor.movedim(-1, 0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (MAX_RESOLUTION * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = F.interpolate(image_tensor, (height, MAX_RESOLUTION), mode='bilinear')
            else:
                width = (MAX_RESOLUTION * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = F.interpolate(image_tensor, (MAX_RESOLUTION, width), mode='bilinear')
            image_tensor = image_tensor.movedim(0, -1)
        previews_ubyte = (((image_tensor + 1.0) / 2.0).clamp(0, 1).mul(0xFF)
                          ).to(device="cpu", dtype=torch.uint8)
        for preview in previews_ubyte:
            i = Image.fromarray(preview.numpy())
            message = BytesIO()
            message.write((1).to_bytes(length=4, byteorder='big') * 2)
            message.write(ind.to_bytes(length=4, byteorder='big'))
            message.write(struct.pack('16p', serv.last_node_id.encode('ascii')))
            i.save(message, format="JPEG", quality=95, compress_level=1)
            serv.send_sync(server.BinaryEventTypes.PREVIEW_IMAGE,
                           message.getvalue(), serv.client_id)
            ind = (ind + 1) % leng

    decode_latent_to_preview_image._h3_patched = True
    process_previews._h3_patched = True
    wrapped_previewer.decode_latent_to_preview_image = decode_latent_to_preview_image
    wrapped_previewer.process_previews = process_previews


def get_h3_tae_previewer(device, latent_format):
    return H3TaePreviewer()

def get_wrapped_h3_tae_previewer(device, latent_format):
    return VHS_WRAPPER(H3TaePreviewer(), rate)

def check_eligibility_to_patch(latent_image, preview_method):
    B, C, T, H, W = latent_image.shape
    if T > 1 and preview_method == "taeh3": # a video (t>1) to be animated
        if VHS_WRAPPER: # if installed, patch
            extra = next(iter(server.PromptServer.instance.prompt_queue.currently_running.values()))[3]['extra_pnginfo']['workflow']['extra']
            active = extra.get('VHS_latentpreview', False)
            if not active:
                print(f"[ComfyUI-GTNodes] VHS animated latentpreview toggle: OFF, the preview won't be animated.\nYou can turn it ON in: Settings -> 🎥🅥🅗🅢 -> \"Display animated previews when sampling\"")
                latent_preview.get_previewer = get_h3_tae_previewer
                return False, None
            
            global rate
            rate = extra.get('VHS_latentpreviewrate', 0)
            print(f"[ComfyUI-GTNodes] Patching VHS's WrappedPreviewer: {VHS_WRAPPER.__module__}")
            patch_vhs(True, VHS_WRAPPER)
            latent_preview.get_previewer = get_wrapped_h3_tae_previewer
            return True, VHS_WRAPPER
        else:
            print("[ComfyUI-GTNodes] VHS not found, the preview won't be animated")
            latent_preview.get_previewer = get_h3_tae_previewer
    elif T == 1 and preview_method == "taeh3": # 1 frame (t==1)
        latent_preview.get_previewer = get_h3_tae_previewer
    return False, None
