def run_step(current_tokens, drafter, verifier, step_idx, total_steps, **kwargs):
    # For Fast-dLLM, we typically let the model manage the steps internally 
    # as it's a block-diffusion pipeline.
    
    ver_out = verifier.generate(
        input_toks={"input_ids": current_tokens},
        max_new_tokens=kwargs.get("gen_length", 256),
        small_block_size=8,
        threshold=0.9
    )
    
    return {
        "next_tokens": ver_out["tokens"],
        "metrics": {"verifier_nfe": 1, "drafter_nfe": 0}
    }