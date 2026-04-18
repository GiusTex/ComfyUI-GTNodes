# ComfyUI-MoreEfficientSamplers
A ksampler inspired by efficiency-nodes comfyui bundling some native nodes.

### Updates:
- Updated `SamplerCustomAdvancedEfficient` to work through comfyui `Sampler Custom Advanced` to support LTXV combined latent and future comfyui updates.
- Added `ScheduledCfgGuider`, `CFGFloatListScheduler` (taken from my Wan-TimeToMove) to support higher cfg values at first steps without needing Wan-TimeToMove.
- Removed redundant nodes.

<img width="259" height="655" alt="SamplerCustomAdvancedEfficient" src="https://github.com/user-attachments/assets/dc348951-35a5-456a-9940-fe22c96f8af0" />

An edit of efficiency-nodes's `ksampler adv. (eff.)` to let it accept custom samplers, schedulers (sigmas) and guiders through comfyui `Sampler Custom Advanced` and other native nodes.
Supports:
- internal slicing of the sigmas, useful for multi-stage sampling;
- use of only positive, useful for sampling with cfg 1;
- use of optional custom samplers, scheduler and guiders. Connect and use them or bypass them/leave the slots empty to use default widgets.

<img width="378" height="221" alt="Scheduled Cfg Guider" src="https://github.com/user-attachments/assets/821528e5-49b3-4016-bce9-b6d49f885136" />

### Suggested resources:
- [`flowmatch scheduler`](https://github.com/BigStationW/flowmatch_scheduler-comfyui): useful with video models using lightx loras;
- `ScheduledCfgGuider`: supports higher cfg values at first steps.

### Download
To install ComfyUI-MoreEfficientSamplers, follow these steps:
- Go in the ComfyUI `custom_nodes` folder, then download the repository or clone it here: `git clone https://github.com/GiusTex/ComfyUI-MoreEfficientSamplers.git`.
- Restart ComfyUI.
