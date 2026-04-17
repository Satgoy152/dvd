import torch
from src.utils import sample_verifier

def run_step(current_tokens, drafter, verifier, step_idx, total_steps, **kwargs):
    """
    Pure Diffusion Baseline for Dream.
    The adapter does a single forward pass, and sample_verifier handles the denoising.
    """
    gen_length = int(kwargs.get("gen_length", 256))
    
    # Dynamically fetch the mask ID (this ensures we use 151670 for Dream, not LLaMA's mask)
    mask_id = verifier.model_state["tokenizer"].mask_token_id

    # 1. Pure Forward Pass (The adapter ignores steps/gen_length now)
    ver_out = verifier.generate(
        input_toks={"input_ids": current_tokens}
    )
    
    # 2. Denoise and Update Tokens
    next_tokens = sample_verifier(
        logits=ver_out["logits"], 
        tokens=current_tokens, 
        mask_id=mask_id, 
        gen_length=gen_length, 
        steps=total_steps
    )
    
    return {
        "next_tokens": next_tokens,
        "metrics": {
            "verifier_nfe": 1,
            "drafter_nfe": 0
        }
    }