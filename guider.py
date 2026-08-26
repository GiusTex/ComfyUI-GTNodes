import comfy
import torch

from comfy.samplers import CFGGuider, sampling_function, process_conds


# Custom Guider, made for Time-To-Move, that accepts a list of cfg values too (cfg values part taken from Kijai WanVideo-Wrapper)
class ScheduledCfgGuiderClass(CFGGuider):
    def set_ttm_options(self, ttm_options):
        self.start_sampler_step = ttm_options["start_sampler_step"]

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        #---------------------------------------------------------
        sigmas = model_options["sigmas"]
        i = torch.argmin(torch.abs(sigmas - timestep)).item()
        return sampling_function(self.inner_model, x, timestep, 
                                 self.conds.get("negative", None), 
                                 self.conds.get("positive", None), 
                                 self.cfg[i],
                                 model_options=model_options, seed=seed)

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}
        
        #---------------------------------------------------------
        skipped_sigmas = sigmas[self.start_sampler_step:]
              # 4   <    5
        if len(skipped_sigmas) < len(sigmas): # sampler doesn't have start_step option
            sigmas = skipped_sigmas
              # 4   ==   4
        elif len(skipped_sigmas) == len(sigmas): # sampler already has option
            pass # we don't want another sigma less
        steps = len(sigmas)-1
        #---------------------------------------------------------
        # Pass sigmas to KSAMPLER.sample
        extra_args["model_options"]["sigmas"] = sigmas
        #---------------------------------------------------------
        # Cfg schedule taken from Kijai WanVideo-Wrapper
        if isinstance(self.cfg, list):
            if steps < len(self.cfg):
                print(f"Received {len(self.cfg)} cfg values, but only {steps} steps. Slicing cfg list to match steps.")
                self.cfg = self.cfg[:steps]
            elif steps > len(self.cfg):
                print(f"Received only {len(self.cfg)} cfg values, but {steps} steps. Extending cfg list to match steps.")
                self.cfg.extend([self.cfg[-1]] * (steps - len(self.cfg)))
            print(f"Using per-step cfg list: {self.cfg}")
        else:
            self.cfg = [self.cfg] * (steps + 1)
        #---------------------------------------------------------

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
        )
        
        # run steps and get final samples
        samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        
        return self.inner_model.process_latent_out(samples.to(torch.float32))


class ScheduledCfgGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Works with a list of floats too (one cfg float per step)"}),
                    "start_sampler_step": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Start step of the whole sampling process. It will automatically skip the selected number of sigmas (starting from the first ones); if the sampler has a start_step option and you changed its value, set the same here"}),
                    },
                }

    DEPRECATED = True
    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "guide"
    CATEGORY = "More Efficient Samplers"
    DESCRIPTION = "Deprecated, it still works but it's better if you use comfyui \"CFG Override\" node.\m\"Scheduled Cfg Guider\": A guider that accepts also a list of cfg values, useful if you want to give at the first step higher ones."

    def guide(cls, model, positive, negative, cfg, start_sampler_step):
        guider = ScheduledCfgGuiderClass(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        ttm_options = {}
        ttm_options["start_sampler_step"] = start_sampler_step
        guider.set_ttm_options(ttm_options)

        return (guider,)

# Taken from kijai WanVideo-Wrapper
class CFGFloatListScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 30, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of steps to schedule cfg for"} ),
            "cfg_scale_start": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "cfg_scale_end": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "interpolation": (["linear", "ease_in", "ease_out"], {"default": "linear", "tooltip": "Interpolation method to use for the cfg scale"}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "Start percent of the steps to apply cfg"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "End percent of the steps to apply cfg"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("float_list",)
    FUNCTION = "process"
    CATEGORY = "More Efficient Samplers"
    DESCRIPTION = "Helper node to generate a list of floats that can be used to schedule cfg scale for the steps, outside the set range cfg is set to 1.0. Taken from Kijai WanVideo-Wrapper"

    def process(self, steps, cfg_scale_start, cfg_scale_end, interpolation, start_percent, end_percent, unique_id):

        # Create a list of floats for the cfg schedule
        cfg_list = [1.0] * steps
        start_idx = min(int(steps * start_percent), steps - 1)
        end_idx = min(int(steps * end_percent), steps - 1)

        for i in range(start_idx, end_idx + 1):
            if i >= steps:
                break

            if end_idx == start_idx:
                t = 0
            else:
                t = (i - start_idx) / (end_idx - start_idx)

            if interpolation == "linear":
                factor = t
            elif interpolation == "ease_in":
                factor = t * t
            elif interpolation == "ease_out":
                factor = t * (2 - t)

            cfg_list[i] = round(cfg_scale_start + factor * (cfg_scale_end - cfg_scale_start), 2)

        # If start_percent > 0, always include the first step
        if start_percent > 0:
            cfg_list[0] = 1.0

        return (cfg_list,)
