# GTNodes for ComfyUI
A bundle of nodes with different purposes. More info below. Some of those nodes (`SamplerCustomAdvancedEfficient`, `MultiLoraLoader`, ...) exist as an alternative to comfyui's subgraphs, where sometimes the widget values are not passed correctly inside. Some nodes use parts of other auhors nodes, more info below/in the credits.

### Updates:
- 8/11/2026: Added `taeh3` preview method and `H3 Tiled Sampler Custom Advanced` node.
- 31/08/2026: Added `MultiLoraLoader` and `MultiLoraLoaderAdvanced`.
- 26/08/2026:
  - Updated `SamplerCustomAdvancedEfficient`: model input is now optional. Added audio decode option. Added option to disable cfg and negative inputs to slightly slim down the ui.
  - Linked `ScheduledCfgGuiderClass` to `CFGGuider`, and removed now redundant functions.
- Updated `SamplerCustomAdvancedEfficient` to work through comfyui `Sampler Custom Advanced` to support LTXV combined latent and future comfyui updates.
- Added `ScheduledCfgGuider`, `CFGFloatListScheduler` (taken from my Wan-TimeToMove) to support higher cfg values at first steps without needing Wan-TimeToMove.
- Removed redundant nodes.

### Download
To install ComfyUI-GTNodes, follow these steps:
- Go in the ComfyUI `custom_nodes` folder, then download the repository or clone it here: `git clone https://github.com/GiusTex/ComfyUI-GTNodes`.
- Optional: install [ComfyUI-SigmaSync](https://github.com/capitan01R/ComfyUI-SigmaSync-LoRA#installation) if you want to use `Multi Lora Loader Advanced`.
- Restart ComfyUI.

### Nodes:
**Multi Lora Loader**

<img width="209" height="139" alt="Multi Lora Loader" src="https://github.com/user-attachments/assets/8f33a184-d276-49f3-ad15-8a4f58e959ea" />

- A simple wrapper around comfyui's `Load Lora` node to load multiple LoRAs.
- **Warning**: changing number of loras resets the values, I suggest to start with 3-4 loras and leave `None` in the unused slots.
- As mentioned above, widgets with `None` are skipped.

**Multi Lora Loader Advanced**

<img width="747" height="243" alt="Multi Lora Loader Advanced" src="https://github.com/user-attachments/assets/dd658b8b-d4c7-4538-9845-bd9d1f3debcb" />

- A wrapper around `Multi Lora Loader` and a simpler version of [`ComfyUI-SigmaSync-LoRA`](https://github.com/capitan01R/ComfyUI-SigmaSync-LoRA#installation), to specify just when to start or end the use of each lora. As you may imagine, the node requires [ComfyUI-SigmaSync](https://github.com/capitan01R/ComfyUI-SigmaSync-LoRA#installation).
- Leave `end_at_step`: `-1` to apply the lora's strength until the last step.
- The sigmas required are the same ones you pass to `SamplerCustomAdvanced`.
- As for `Multi Lora Loader`, widgets with `None` are skipped, and changing number of loras **resets the values**. I suggest to start with number of loras: 3 and leave `None` in the unused slots.

_How it works_:

- Instead of specifying different strengths at different steps or setting manually their list manually, the node forces the [`explicit curve`](https://github.com/GiusTex/ComfyUI-GTNodes/blob/add-multiloraloader-and-multiloraloaderadvanced/nodes/models.py#L138) and builds the `strengths_list` with a list of zeros, then it applies the lora's strength at the specified steps.
- If no sigmas are provided the node falls back to the `Multi Lora Loader` node, otherwise it applies each strenght at specified steps to the corresponding loras.

**Sampler Custom Advanced (Efficient)**:

<img width="255" height="707" alt="image" src="https://github.com/user-attachments/assets/c9226e2f-9a7f-4aea-a01e-f7d1201d92a2" />

An edit of efficiency-nodes's `ksampler adv. (eff.)` to let it accept custom samplers, schedulers (sigmas) and guiders through comfyui `Sampler Custom Advanced` and other native nodes.
Supports:
- internal slicing of the sigmas, useful for multi-stage sampling;
- use of only positive, useful when sampling with cfg 1 and/or the negative isn't required;
- use of optional custom samplers, schedulers and guiders. Connect and use them, or bypass/leave their input slots empty to use the corresponding default widgets. Default guiders: `CFG Guider` when `use_cfg` is `True`, `Basic Guider` when `False`.
- Decoding of images/audio, when one or both are available (the node auto-checks what is available in the latent).
- Embedded preview for MiniMax H3, using `taeh3` preview method. `Max_preview_resolution` is hardcoded at 1024 (instead of 512 like in default ComfyUI, anyway only greater inputs will be resized).

  Requires kijai's [taeh3 vae](https://huggingface.co/Kijai/MiniMax-H3-TAE/blob/main/vae_approx/taeh3.safetensors) in `models/vae_approx`.

  Requires an H3 latent dict (image/video + audio), and works both with a single image or a video. For the single image you can use `P-H3-Image Conditioning` node, that returns a latent dict of image + audio, then their H3 image vae or another custom image vae (link in "Suggested resources").

  For an animated video preview, Kosinkadink/[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) is also required, then from ComfyUI's Settings window -> 🎥🅥🅗🅢: toggle "Display animated previews when sampling" ON. You can also change video preview's fps here.

  Decoding the preview stops the sampler for some seconds. Decode tests (on a rtx 3060 12gb):

  - video of 8 seconds, size of 448x704: the sampler stopped each step for 1.3 seconds;
  - video of 8 seconds, size of 576x832: the sampler stopped each step for 2.1 seconds.

**H3 Tiled Sampler Custom Advanced**:

<img width="244" height="717" alt="image" src="https://github.com/user-attachments/assets/cb23a9f4-d301-4917-9f17-caee8a1bd2d5" />

An edit of was-node-suite-comfyui's `H3 Tiled Sampler`, to let it accept custom sigmas, useful if you have lightx loras that require custom sigmas, like MiniMax H3 HyperFlow model/lora.

Requires an H3 latent dict (image/video + audio), and works both with a single image or a video, check `Sampler Custom Advanced (Efficient)` for more info.

To use `taeh3` previouw method, as said in the previous node, kijai's [taeh3 vae](https://huggingface.co/Kijai/MiniMax-H3-TAE/blob/main/vae_approx/taeh3.safetensors) is required in `models/vae_approx`; more info above on this and "Display animated previews when sampling" (to turn on animated video preview).

This tiled sampler is useful if you upscaled an image/video, for example with the ComfyUI's native `MiniMax H3 Latent Upscale With Model` node (example upscale model in "Suggested resources"), the sampler then returns a dict, so you have to use `Separate AV Latent` node to get the video.

**Scheduled CFG Guider** (with its **CFG Float List Scheduler**):

<img width="378" height="221" alt="Scheduled Cfg Guider" src="https://github.com/user-attachments/assets/821528e5-49b3-4016-bce9-b6d49f885136" />

(*Deprecated*, although they still work; I advice to use comfyui's native `CFG Override`). Scheduled CFG Guider, anyway, let you decide what cfg to apply at each step through a list of floats connected to the cfg input. Based on kijai nodes.

### Suggested resources:
- [`flowmatch scheduler`](https://github.com/BigStationW/flowmatch_scheduler-comfyui): useful with video models using lightx loras;
- [P-H3-Image-Comfy's vae](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/blob/main/vae/p-h3-image_vae_int8_convrot.safetensors);
- custom MiniMax H3 [image vae](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/blob/main/minimax_h3_t1_image_vae_step1597.safetensors);
- [minimax_h3_latent_upscaler_3d](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/09592c6221ec95cc8e0fae67842e34926c4e668b/minimax_h3_latent_upscaler_3d_bf16.safetensors).

### Credits:
- jags111/efficiency-nodes-comfyui, whose [`KSampler Adv (Efficient)`](https://github.com/jags111/efficiency-nodes-comfyui/blob/main/efficiency_nodes.py#L2208) was the base for the `Sampler Custom Advanced (Efficient)`;
- WASasquatch/was-node-suite-comfyui, for his [`H3TiledSampler`](https://github.com/WASasquatch/was-node-suite-comfyui/blob/cc9e1fb34155086c094209b8c7793c382b4f8d0a/nodes/sampling/h3_tiled_sampler.py#L78) and his 3 modules (h3_low_vram.py, h3_tiles.py, preview.py) which still fills most of the `H3 Tiled Sampler Custom Advanced`;
- jlucasmcrell/ComfyUI-H3-Multishot, whose [`_build_decoder`](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot/blob/d7d197709fc93c1afb125e3db1dfd4ac0ae28c60/h3_tae_decode.py#L20) is used to build the taeh3 decoder;
- Kosinkadink/ComfyUI-VideoHelperSuite, whose `decode_latent_to_preview_image` and `process_previews` are patched from ([link](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/videohelpersuite/latent_preview.py#L18));
- Claude for helping creating and debugging the code.
