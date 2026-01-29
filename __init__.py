from .samplers import (SamplerCustomAdvanced_Efficient,
                       SamplerCustomUltraAdvancedEfficient,
                       SamplerCustomUltraAdvancedPlusEfficient)


NODE_CLASS_MAPPINGS = {
    "SamplerCustomAdvanced_Efficient": SamplerCustomAdvanced_Efficient,
    "SamplerCustomUltraAdvancedEfficient": SamplerCustomUltraAdvancedEfficient,
    "SamplerCustomUltraAdvancedPlusEfficient": SamplerCustomUltraAdvancedPlusEfficient,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerCustomAdvanced_Efficient": "Sampler Custom Advanced (Efficient)",
    "SamplerCustomUltraAdvancedEfficient": "Sampler Custom Ultra Advanced (Efficient)",
    "SamplerCustomUltraAdvancedPlusEfficient": "Sampler Custom Ultra Advanced Plus (Efficient)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
