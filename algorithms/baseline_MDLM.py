from src.utils import sample_verifier
import torch

def run_step(current_tokens, drafter, verifier, step_idx, total_steps, **kwargs):
    gen_length = int(kwargs.get("gen_length"))
    tokenizer = verifier.model_state["tokenizer"]
    mask_id = tokenizer.mask_token_id
    
    # Validation: In MDLM/LLaDA, the sequence length should NOT change between steps.
    # If this print changes, your outer loop is likely appending instead of replacing.
    # print(f"DEBUG: Step {step_idx}, Sequence Length: {current_tokens.shape[1]}")

    # 1. Forward Pass
    # We pass 'steps=step_idx + 1' so the adapter knows this is NOT the first call
    ver_out = verifier.generate(
        input_toks={"input_ids": current_tokens},
        steps=step_idx + 1, 
        gen_length=gen_length
    )
    
    # 2. Sample/Update
    # This modifies current_tokens in-place (or returns a modified clone)
    next_tokens = sample_verifier(
        logits=ver_out["logits"],
        tokens=current_tokens,
        mask_id=mask_id,
        gen_length=gen_length,
        steps=total_steps
    )
    
    return {
        "next_tokens": next_tokens,
        "metrics": {"verifier_nfe": 1, "drafter_nfe": 0}
    }