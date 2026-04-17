from src.utils import sample_verifier
import torch

def run_step(current_tokens, drafter, verifier, step_idx, total_steps, **kwargs):
    """
    A baseline algorithm demonstrating the adapter pattern.
    Here, the verifier simply runs a single forward pass and samples tokens.
    It demonstrates returning the next tokens and custom metrics.
    """
    gen_length = int(kwargs.get("gen_length"))
    mask_id = int(kwargs.get("mask_id", 126336))
    
    # We step the verifier once on the current tokens
    ver_out = verifier.generate(input_toks={"input_ids": current_tokens}, steps=step_idx+1)
    
    # Sample new unmasked tokens
    next_tokens = sample_verifier(
        logits=ver_out["logits"], 
        tokens=current_tokens, 
        mask_id=mask_id, 
        gen_length=gen_length, 
        steps=total_steps
    )
    
    # A generic implementation might also run the drafter, but here we just
    # simulate taking NFE counts so the runtime can aggregate them.
    return {
        "next_tokens": next_tokens,
        "metrics": {
            "verifier_nfe": 1,
            "drafter_nfe": 0
        }
    }
