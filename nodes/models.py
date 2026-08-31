import os
import sys
import importlib

import comfy.utils
import folder_paths
from comfy_api.latest import io

from nodes import LoraLoader


class MultiLoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = []
        for num_loras in range(1, 21):  # 1 to 20 loras
            lora_inputs = []
            for i in range(1, num_loras + 1):
                lora_inputs.extend([
                    io.Combo.Input(f"lora_name_{i}", ["None"] + folder_paths.get_filename_list("loras"), default="None"),
                    io.Float.Input(f"strength_model_{i}", default=1.0, min=0.0, max=10.0, step=0.01, tooltip=f"Strength for lora {i}."),
                ])
            options.append(io.DynamicCombo.Option(
                key=str(num_loras),
                inputs=lora_inputs
            ))

        return io.Schema(
            node_id="MultiLoraLoader",
            display_name="Multi Lora Loader",
            category="ComfyUI-GTNodes/models",
            description="Add multiple loras to the model with strengths, uses DynamicCombo which requires ComfyUI 0.8.1 and frontend 1.33.4 or later.",
            inputs=[
                io.Model.Input("model"),
                io.DynamicCombo.Input("num_loras", options=options, display_name="Number of Loras", tooltip="Select how many loras to use"),
            ],
            outputs=[
                io.Model.Output(display_name="model", tooltip="Model with added loras"),
            ],
        )

    @classmethod
    def execute(cls, model, num_loras) -> io.NodeOutput:

        lora_keys = sorted([k for k in num_loras.keys() if k.startswith('lora_name_')])

        for lora_key in lora_keys:
            i = lora_key.split('_')[2]

            lora_name = num_loras[f"lora_name_{i}"]
            if lora_name != "None":
                strength_model = num_loras[f"strength_model_{i}"]
    
                model = LoraLoader().load_lora(
                    model, None, lora_name, strength_model, 0
                )[0]

        return io.NodeOutput(model)


class SigmaSyncLoraLoaderModelOnlyCustom:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sigmas": ("SIGMAS", {"tooltip": "Sigmas used for the generation, connect them to specify at which step start and/or end to apply the LoRAs.\nRequires https://github.com/capitan01R/ComfyUI-SigmaSync-LoRA"}),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "start_at_step": ("INT", {"default": 0, "min": 0}),
                "end_at_step": ("INT", {"default": -1, "min": -1, "tooltip": "Leave -1 to use same strength accross all steps, otherwise specify the step where this lora's strength becomes 0 (the lora will keep strength 0 until end of generation"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "ComfyUI-GTNodes/models"
    DESCRIPTION = "Loads a model-only LoRA and schedules its runtime strength against the exact SIGMAS tensor supplied to the sampler.\nRequires https://github.com/capitan01R/ComfyUI-SigmaSync-LoRA"

    @classmethod
    def _get_sigmaSync_module(cls):
        """Import Sigma-Sync module with version validation"""
        for key, mod in sys.modules.items():
            if key.endswith("ComfyUI-SigmaSync-LoRA") or key.endswith("comfyui-sigmasync-lora"):
                if hasattr(mod, "sigma_sync_lora"):
                    return mod

        sigmaSync_path = os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0], "ComfyUI-GGUF")
        for module_name in ["ComfyUI-SigmaSync-LoRA", "custom_nodes.ComfyUI-SigmaSync-LoRA", "comfyui-sigmasync-lora", "custom_nodes.comfyui-sigmasync-lora", sigmaSync_path, sigmaSync_path.lower()]:
            try:
                module = importlib.import_module(module_name)
                return module
            except ImportError:
                continue

        raise ImportError(
            "Compatible ComfyUI-SigmaSync-LoRA not found. "
            "Please install/update from: https://github.com/capitan01R/ComfyUI-SigmaSync-LoRA"
        )

    def _load_state(cls, lora_name):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if cls.loaded_lora is None or cls.loaded_lora[0] != lora_path:
            state = comfy.utils.load_torch_file(lora_path, safe_load=True)
            cls.loaded_lora = (lora_path, state)
        return cls.loaded_lora

    def load_lora(cls, model, sigmas, lora_name, strength, start_at_step, end_at_step):
        sigmaSync_nodes = cls._get_sigmaSync_module()
        anchors = sigmaSync_nodes.sigma_sync_lora._sigma_anchors(sigmas)

        # Initialize strength list with zeros
        strengths_list = [0.0] * len(anchors)

        # Apply Lora to all steps
        if end_at_step < 0:
            end_at_step = len(anchors)

        # Apply strength only in desired range
        for i in range(start_at_step, min(end_at_step, len(anchors))):
            strengths_list[i] = strength

        # When only zeros skip Lora
        if all(s == 0.0 for s in strengths_list):
            return (model.clone(),)

        # Load LoRA
        lora_path, state = cls._load_state(lora_name)
        spec = {
            "lora_path": lora_path,
            "state": state,
            "sigma_anchors": anchors,
            "strength_anchors": strengths_list,
            "curve": "explicit", # We are forcing this method
        }

        updated_m = sigmaSync_nodes.sigma_sync_lora._extend_model(model, spec)

        return (updated_m,)


class MultiLoraLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = []
        for num_loras in range(1, 21):  # 1 to 20 loras
            lora_inputs = []
            for i in range(1, num_loras + 1):
                lora_inputs.extend([
                    io.Combo.Input(f"lora_name_{i}", ["None"] + folder_paths.get_filename_list("loras"), default="None"),
                    io.Float.Input(f"strength_model_{i}", default=1.0, min=0.0, max=10.0, step=0.01, tooltip=f"Strength for lora {i}."),
                    io.Int.Input(f"start_lora_{i}_at_step", default=0, min=0, tooltip="Requires sigmas"),
                    io.Int.Input(f"end_lora_{i}_at_step", default=-1, min=-1, tooltip="Requires sigmas. Leave -1 to use same strength accross all steps, otherwise specify at which step to stop applying this LoRA"),
                ])
            options.append(io.DynamicCombo.Option(
                key=str(num_loras),
                inputs=lora_inputs
            ))

        return io.Schema(
            node_id="MultiLoraLoaderAdvanced",
            display_name="Multi Lora Loader Advanced",
            category="ComfyUI-GTNodes/models",
            description="Add multiple loras to the model with strengths, uses DynamicCombo which requires ComfyUI 0.8.1 and frontend 1.33.4 or later.\nRequires https://github.com/capitan01R/ComfyUI-SigmaSync-LoRA is using lora scheduling",
            inputs=[
                io.Model.Input("model"),
                io.Sigmas.Input("sigmas", optional=True, tooltip="Sigmas used for the generation, connect them to specify at which step start and/or end to apply the LoRAs.\nRequires https://github.com/capitan01R/ComfyUI-SigmaSync-LoRA"),
                io.DynamicCombo.Input("num_loras", options=options, display_name="Number of LoRAs", tooltip="Select how many loras to use"),
            ],
            outputs=[
                io.Model.Output(display_name="model", tooltip="Model with added loras"),
            ],
        )

    @classmethod
    def execute(cls, model, num_loras, sigmas=None) -> io.NodeOutput:

        lora_keys = sorted([k for k in num_loras.keys() if k.startswith('lora_name_')])

        for lora_key in lora_keys:
            i = lora_key.split('_')[2]

            lora_name = num_loras[f"lora_name_{i}"]
            if lora_name != "None":
                strength = num_loras[f"strength_model_{i}"]
                start_at_step = num_loras[f"start_lora_{i}_at_step"]
                end_at_step = num_loras[f"end_lora_{i}_at_step"]

                if sigmas is not None:
                    model = SigmaSyncLoraLoaderModelOnlyCustom().load_lora(
                        model, sigmas, lora_name, strength, start_at_step, end_at_step
                    )[0]
                else:
                    model = LoraLoader().load_lora(
                        model, None, lora_name, strength, 0
                    )[0]

        return io.NodeOutput(model)


MODELS_CLASS_MAPPINGS = {
    "MultiLoraLoader": MultiLoraLoader,
    #"SigmaSyncLoraLoaderModelOnlyCustom": SigmaSyncLoraLoaderModelOnlyCustom,
    "MultiLoraLoaderAdvanced": MultiLoraLoaderAdvanced,
}

MODELS_NAME_MAPPINGS = {
    "MultiLoraLoader": "Multi Lora Loader",
    #"SigmaSyncLoraLoaderModelOnlyCustom": "SigmaSync LoraLoader ModelOnly Custom",
    "MultiLoraLoaderAdvanced": "Multi Lora Loader Advanced",
}
