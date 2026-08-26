# GTNodes for ComfyUI
A bundle of nodes with different purposes. More info below.

### Updates:
- 26/08/2026:
  - Updated `SamplerCustomAdvancedEfficient`: model input is now optional. Added audio decode option. Added option to disable cfg and negative inputs to slightly slim down the ui.
  - Linked `ScheduledCfgGuiderClass` to `CFGGuider`, and removed now redundant functions.
- Updated `SamplerCustomAdvancedEfficient` to work through comfyui `Sampler Custom Advanced` to support LTXV combined latent and future comfyui updates.
- Added `ScheduledCfgGuider`, `CFGFloatListScheduler` (taken from my Wan-TimeToMove) to support higher cfg values at first steps without needing Wan-TimeToMove.
- Removed redundant nodes.

### Download
To install ComfyUI-GTNodes, follow these steps:
- Go in the ComfyUI `custom_nodes` folder, then download the repository or clone it here: `git clone https://github.com/GiusTex/ComfyUI-GTNodes`.
- Restart ComfyUI.

### Nodes:
**Sampler Custom Advanced (Efficient)**:

<img width="320" height="513" alt="SamplerCustomAdvancedEfficient_Updated" src="https://github.com/user-attachments/assets/e8016887-3070-440e-af65-9e038dbb2fc1" />

An edit of efficiency-nodes's `ksampler adv. (eff.)` to let it accept custom samplers, schedulers (sigmas) and guiders through comfyui `Sampler Custom Advanced` and other native nodes.
Supports:
- internal slicing of the sigmas, useful for multi-stage sampling;
- use of only positive, useful when sampling with cfg 1 and/or the negative isn't required;
- use of optional custom samplers, schedulers and guiders. Connect and use them, or bypass/leave their input slots empty to use the corresponding default widgets. Default guiders: `CFG Guider` when `use_cfg` is `True`, Basic Guider when false.

**Scheduled CFG Guider** (with its **CFG Float List Scheduler**):

<img width="378" height="221" alt="Scheduled Cfg Guider" src="https://github.com/user-attachments/assets/821528e5-49b3-4016-bce9-b6d49f885136" />
- *Deprecated*, although they still work; I advice to use comfyui native `CFG Override`. Scheduled CFG Guider, anyway, let you decide what cfg to apply at each step through a list of floats connected to the cfg input. Based on kijai nodes.

### Suggested resources:
- [`flowmatch scheduler`](https://github.com/BigStationW/flowmatch_scheduler-comfyui): useful with video models using lightx loras;
- `ScheduledCfgGuider`: supports higher cfg values at first steps.
