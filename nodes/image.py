import torch
import numpy as np
from PIL import Image, ImageDraw

from comfy_api.latest import io


# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True


class PadImageGT(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PadImageGT",
            description="Standalone Pad Image node from ComfyUI-DiffusersImageOutpaint",
            display_name="Pad Image GT",
            category="ComfyUI-GTNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=720, tooltip="The width used for the image"),
                io.Int.Input("height", default=1280, tooltip="The height used for the image"),
                io.Combo.Input("alignment", ["Middle", "Left", "Right", "Top", "Bottom"], default="Middle", tooltip="Where the original image should be in the outpainted one"),
                io.Combo.Input("resize_image", ["Full", "50%", "33%", "25%", "Custom"], default="Full", tooltip="Resize input image"),
                io.Int.Input("custom_resize_image_percentage", default=50, min=1, max=100, step=1, tooltip="Custom resize (%)"),
                io.Int.Input("mask_overlap_percentage", default=10, min=1, max=50, step=1, tooltip="Mask overlap (%)"),
                io.Boolean.Input("overlap_left", default=True),
                io.Boolean.Input("overlap_right", default=True),
                io.Boolean.Input("overlap_top", default=True),
                io.Boolean.Input("overlap_bottom", default=True),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.Image.Output(display_name="masked image for outpaint"), 
            ],
        )
    
    @classmethod
    def execute(cls, image, width, height, mask_overlap_percentage, resize_image, custom_resize_image_percentage, overlap_left, overlap_right, overlap_top, overlap_bottom, alignment="Middle"):
        im=tensor2pil(image)
        source=im.convert('RGB')

        target_size = (width, height)
        
        # Calculate the scaling factor to fit the image within the target size
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)

        # Resize the source image to fit within target size
        source = source.resize((new_width, new_height), Image.LANCZOS)

        # Initialize new_width and new_height
        new_width, new_height = source.width, source.height

        # Apply resize option using percentages
        if resize_image == "Full":
            resize_percentage = 100
        elif resize_image == "50%":
            resize_percentage = 50
        elif resize_image == "33%":
            resize_percentage = 33
        elif resize_image == "25%":
            resize_percentage = 25
        else:  # Custom
            resize_percentage = custom_resize_image_percentage
        
        # Calculate new dimensions based on percentage
        resize_factor = resize_percentage / 100
        new_width = int(source.width * resize_factor)
        new_height = int(source.height * resize_factor)
        
        # Ensure minimum size of 64 pixels
        new_width = max(new_width, 64)
        new_height = max(new_height, 64)

        # Resize the image
        source = source.resize((new_width, new_height), Image.LANCZOS)

        # Calculate the overlap in pixels based on the percentage
        overlap_x = int(new_width * (mask_overlap_percentage / 100))
        overlap_y = int(new_height * (mask_overlap_percentage / 100))
        
        # Ensure minimum overlap of 1 pixel
        overlap_x = max(overlap_x, 1)
        overlap_y = max(overlap_y, 1)
        
        # Calculate margins based on alignment
        if alignment == "Middle":
            margin_x = (target_size[0] - source.width) // 2
            margin_y = (target_size[1] - source.height) // 2
        elif alignment == "Left":
            margin_x = 0
            margin_y = (target_size[1] - source.height) // 2
        elif alignment == "Right":
            margin_x = target_size[0] - source.width
            margin_y = (target_size[1] - source.height) // 2
        elif alignment == "Top":
            margin_x = (target_size[0] - source.width) // 2
            margin_y = 0
        elif alignment == "Bottom":
            margin_x = (target_size[0] - source.width) // 2
            margin_y = target_size[1] - source.height

        # Adjust margins to eliminate gaps
        margin_x = max(0, min(margin_x, target_size[0] - new_width))
        margin_y = max(0, min(margin_y, target_size[1] - new_height))

        # Create a new background image and paste the resized source image
        background = Image.new('RGB', target_size, (255, 255, 255))
        background.paste(source, (margin_x, margin_y))

        image=pil2tensor(background)
        #----------------------------------------------------
        # Create the mask
        d1, d2, d3, d4 = image.size()
        left, top, bottom, right = 0, 0, 0, 0
        # Image
        new_image = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        ) * 0.5
        new_image[:, top:top + d2, left:left + d3, :] = image

        im=tensor2pil(new_image)
        pil_new_image=im.convert('RGB')
        #----------------------------------------------------

        # Create the mask
        mask = Image.new('L', target_size, 255)
        mask_draw = ImageDraw.Draw(mask)
        #----------------------------------------------------
        # Calculate overlap areas
        white_gaps_patch = 2

        left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
        top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
        #----------------------------------------------------
        # Mask coordinates
        if alignment == "Left":
            left_overlap = margin_x + overlap_x if overlap_left else margin_x
        elif alignment == "Right":
            right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
        elif alignment == "Top":
            top_overlap = margin_y + overlap_y if overlap_top else margin_y
        elif alignment == "Bottom":
            bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height
    
        # Draw the mask
        mask_draw.rectangle([
            (left_overlap, top_overlap),
            (right_overlap, bottom_overlap)
        ], fill=0)

        tensor_mask=pil2tensor(mask)
        #----------------------------------------------------
        if not can_expand(background.width, background.height, width, height, alignment):
            alignment = "Middle"
        
        cnet_image = pil_new_image.copy() # copy background as cnet_image
        cnet_image.paste(0, (0, 0), mask) # paste mask over cnet_image, cropping it a bit
        
        tensor_cnet_image=pil2tensor(cnet_image)

        return io.NodeOutput(new_image, tensor_mask, tensor_cnet_image)


IMAGE_CLASS_MAPPINGS = {
    "PadImageGT": PadImageGT,
}

IMAGE_NAME_MAPPINGS = {
    "PadImageGT": "Pad Image GT",
}
