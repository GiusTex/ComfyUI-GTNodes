# GTNodes for ComfyUI
A bundle of nodes with different purposes. More info below. Some of those nodes (`SamplerCustomAdvancedEfficient`, `MultiLoraLoader`, ...) exist as an alternative to comfyui's subgraphs, where sometimes the widget values are not passed correctly inside. Some nodes use parts of other auhors nodes, more info below and in the credits.

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
- Decoding of images/audio, when one or both are available (the node auto-checks whats available in the latent).

**H3 Tiled Sampler Custom Advanced**:

<img width="244" height="717" alt="image" src="https://github.com/user-attachments/assets/cb23a9f4-d301-4917-9f17-caee8a1bd2d5" />

An edit of...

- Text...

**Scheduled CFG Guider** (with its **CFG Float List Scheduler**):

<img width="378" height="221" alt="Scheduled Cfg Guider" src="https://github.com/user-attachments/assets/821528e5-49b3-4016-bce9-b6d49f885136" />

(*Deprecated*, although they still work; I advice to use comfyui's native `CFG Override`). Scheduled CFG Guider, anyway, let you decide what cfg to apply at each step through a list of floats connected to the cfg input. Based on kijai nodes.

### Suggested resources:
[`flowmatch scheduler`](https://github.com/BigStationW/flowmatch_scheduler-comfyui): useful with video models using lightx loras.

### Credits:
- Efficient nodes...
- was nodes suit...
- jlucasmcrell/ComfyUI-H3-Multishot to build the taeh3 decoder (link: [`_build_decoder`](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot/blob/d7d197709fc93c1afb125e3db1dfd4ac0ae28c60/h3_tae_decode.py#L20));
- video-helper-suite...
- Claude...
