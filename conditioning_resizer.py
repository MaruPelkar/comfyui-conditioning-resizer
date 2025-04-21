import torch
import numpy as np
from comfy.sd3 import T5Tokenizer, FluxModelBase
from comfy import model_management

class ConditioningResizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "target_length": ("INT", {"default": 5273, "min": 1, "max": 10000}),
                "resize_method": (["pad_or_trim", "interpolate"], {"default": "pad_or_trim"}),
                "pad_value": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "resize_conditioning"
    CATEGORY = "conditioning"

    def resize_conditioning(self, conditioning, target_length, resize_method="pad_or_trim", pad_value=0.0):
        result = []
        
        for cond in conditioning:
            c = cond[0]
            
            # Check if we're dealing with conditioning from CLIP Text Encode
            if isinstance(c, dict) and "pooled_output" in c and "cond_pooled" in c and "cross_attn_kwargs" in c:
                resized_cond = c.copy()
                
                # Handle cross_attn_kwargs
                if "cross_attn_kwargs" in c and c["cross_attn_kwargs"] is not None:
                    cross_attn = c["cross_attn_kwargs"].copy()
                    
                    # Handle attention bias resize
                    if "attention_bias" in cross_attn and cross_attn["attention_bias"] is not None:
                        attn_bias = cross_attn["attention_bias"]
                        
                        # Get shape information
                        batch_size, num_heads, seq_len, _ = attn_bias.shape
                        
                        if resize_method == "pad_or_trim":
                            # Create new tensor with target dimensions
                            new_attn_bias = torch.full((batch_size, num_heads, target_length, target_length), 
                                                       pad_value, dtype=attn_bias.dtype, device=attn_bias.device)
                            
                            # Copy as much as possible from the original
                            copy_len = min(seq_len, target_length)
                            new_attn_bias[:, :, :copy_len, :copy_len] = attn_bias[:, :, :copy_len, :copy_len]
                            
                        elif resize_method == "interpolate":
                            # Use interpolation to resize
                            temp = attn_bias.reshape(batch_size * num_heads, 1, seq_len, seq_len)
                            temp = torch.nn.functional.interpolate(
                                temp, size=(target_length, target_length), mode='bilinear', align_corners=False
                            )
                            new_attn_bias = temp.reshape(batch_size, num_heads, target_length, target_length)
                        
                        cross_attn["attention_bias"] = new_attn_bias
                    
                    resized_cond["cross_attn_kwargs"] = cross_attn
                
                result.append([resized_cond, cond[1]])
            else:
                # If not a CLIP Text Encode conditioning, pass through unchanged
                result.append(cond)
                
        return (result,)

# Add the node to the node list
NODE_CLASS_MAPPINGS = {
    "ConditioningResizer": ConditioningResizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningResizer": "Conditioning Resizer",
}
