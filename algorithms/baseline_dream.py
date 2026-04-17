import torch

from algorithms._common import (
    initialize_content,
    step_generate,
    sample_topk_confident,
    decode_results,
)

def run_step(drafter, verifier, prompts, step_idx, total_steps, algo_state, **kwargs):
    """
    Pure Baseline diffusion: Each step unmasks the top-K most confident tokens 
    based on a single forward pass. Dynamically fetches the mask ID to support 
    Dream (151670), LLaDA (126336), or any other MDLM.
    """
    gen_length = int(kwargs.get("gen_length"))
    
    # --- THE DYNAMIC FIX ---
    # We fetch the mask ID directly from the verifier's tokenizer state.
    tokenizer = verifier.model_state["tokenizer"]
    mask_id = tokenizer.mask_token_id
    
    if mask_id is None:
        raise ValueError("[ERROR] Tokenizer has no mask_token_id. Ensure your adapter adds it.")

    if gen_length % total_steps != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be divisible by total_steps ({total_steps})"
        )
    tokens_per_step = gen_length // total_steps

    # Filter kwargs to pass down cleanly
    kwargs_pass = {k: v for k, v in kwargs.items() if k not in ["gen_length", "mask_id"]}
    
    if step_idx == 0:
        # Step 0: Create the initial canvas using the dynamic mask_id
        full_content, full_attn, prompt_len, logits = initialize_content(
            verifier, prompts, gen_length, mask_id, **kwargs_pass
        )
    else:
        # Subsequent steps: Pull from state and run the pure forward pass
        full_content = algo_state["content"]
        full_attn = algo_state["content_attn"]
        prompt_len = algo_state["prompt_len"]
        
        logits = step_generate(
            verifier, full_content, full_attn, prompt_len, gen_length, mask_id
        )

    # Denoise: Select top-K tokens
    gen_region = full_content[:, prompt_len:]
    updated_gen = sample_topk_confident(logits, gen_region, mask_id, tokens_per_step)
    full_content = torch.cat([full_content[:, :prompt_len], updated_gen], dim=1)

    result = {
        "algo_state": {
            "content": full_content,
            "content_attn": full_attn,
            "prompt_len": prompt_len,
        },
        "metrics": {"verifier_nfe": 1, "drafter_nfe": 0},
    }

    # Final Step: Decode the finished sequence
    if step_idx == total_steps - 1:
        result["results"] = decode_results(verifier, full_content, prompts, prompt_len)

    return result