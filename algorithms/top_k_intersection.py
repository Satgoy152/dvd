import torch
from algorithms._common import (
    initialize_content,
    step_generate,
    sample_topk_confident,
    decode_results,
)

def run_step(drafter, verifier, prompts, step_idx, total_steps, algo_state, **kwargs):
    gen_length = int(kwargs.get("gen_length"))
    tokens_per_step = max(1, gen_length // total_steps)

    # 1. DYNAMIC TOKENIZER SETUP
    d_tok = drafter.model_state["tokenizer"]
    v_tok = verifier.model_state["tokenizer"]
    
    d_mask = d_tok.mask_token_id
    if d_mask is None: d_mask = int(kwargs.get("mask_id", 126336))
        
    v_mask = v_tok.mask_token_id
    if v_mask is None: v_mask = int(kwargs.get("mask_id", 126336))

    kwargs_pass = {k: v for k, v in kwargs.items() if k not in ["gen_length", "mask_id"]}
    
    # 2. INITIALIZATION
    if step_idx == 0:
        full_content, full_attn, prompt_len, drafter_logits = initialize_content(
            drafter, prompts, gen_length, d_mask, **kwargs_pass
        )
    else:
        full_content = algo_state["content"]
        full_attn = algo_state["content_attn"]
        prompt_len = algo_state["prompt_len"]
        
        if not (full_content[:, prompt_len:] == d_mask).any():
            return {"algo_state": algo_state, "metrics": {"done_early": True}, "done": True}

        drafter_logits = step_generate(
            drafter, full_content, full_attn, prompt_len, gen_length, d_mask
        )

    gen_region = full_content[:, prompt_len:]

    # 3. DRAFTER PROPOSAL
    drafted_gen = sample_topk_confident(drafter_logits, gen_region, d_mask, tokens_per_step)
    drafted_positions = (gen_region == d_mask) & (drafted_gen != d_mask)

    # 4. PREPARE TENSOR FOR VERIFIER (NO LENGTH CHANGES)
    v_vocab_size = verifier.model_state["model"].config.vocab_size
    ver_input_gen = drafted_gen.clone()
    
    # Swap Drafter mask to Verifier mask
    ver_input_gen = torch.where(ver_input_gen == d_mask, v_mask, ver_input_gen)

    # Reconstruct the full sequence (Prompt + Generation)
    drafted_content_v_space = torch.cat([full_content[:, :prompt_len], ver_input_gen], dim=1)
    
    # Clamp the ENTIRE sequence (including Dream's prompt tokens)
    drafted_content_v_space = torch.clamp(drafted_content_v_space, 0, v_vocab_size - 1)

    # 5. VERIFIER FORWARD PASS
    ver_logits = step_generate(
        verifier, drafted_content_v_space, full_attn, prompt_len, gen_length, v_mask
    )
    ver_predicted_ids = ver_logits.argmax(dim=-1)

    # 6. TRANSLATE MASKS BACK TO DRAFTER SPACE FOR EXACT SHAPE COMPARISON
    ver_predicted_ids_d_space = ver_predicted_ids[:, prompt_len:].clone()
    ver_predicted_ids_d_space = torch.where(ver_predicted_ids_d_space == v_mask, d_mask, ver_predicted_ids_d_space)

    # 7. AGREEMENT & OVERRIDE
    agreed_positions = drafted_positions & (ver_predicted_ids_d_space == drafted_gen)

    next_gen = gen_region.clone()
    next_gen[agreed_positions] = drafted_gen[agreed_positions]

    drafted_per_row = drafted_positions.sum(dim=-1)
    agreed_per_row = agreed_positions.sum(dim=-1)
    failed_per_row = drafted_per_row - agreed_per_row
    
    max_failed = int(failed_per_row.max().item())
    if max_failed > 0:
        remaining_mask = next_gen == d_mask
        ver_confidence = ver_logits.max(dim=-1).values[:, prompt_len:]
        ver_confidence = ver_confidence.masked_fill(~remaining_mask, float("-inf"))
        
        _, topk_indices = ver_confidence.topk(max_failed, dim=-1)
        
        row_range = torch.arange(max_failed, device=topk_indices.device).unsqueeze(0)
        valid_row_mask = row_range < failed_per_row.unsqueeze(1)
        
        # Pull Verifier selections and clamp them safely back into Drafter's vocabulary limit
        d_vocab_size = drafter.model_state["model"].config.vocab_size
        ver_selections = ver_predicted_ids_d_space.gather(1, topk_indices)
        ver_selections = torch.clamp(ver_selections, 0, d_vocab_size - 1)
        
        current_vals = next_gen.gather(1, topk_indices)
        scatter_src = torch.where(valid_row_mask, ver_selections, current_vals)
        next_gen.scatter_(1, topk_indices, scatter_src)

    next_full_content = torch.cat([full_content[:, :prompt_len], next_gen], dim=1)

    result = {
        "algo_state": {
            "content": next_full_content,
            "content_attn": full_attn,
            "prompt_len": prompt_len,
        },
        "metrics": {
            "drafter_nfe": 1,
            "verifier_nfe": 1,
            "drafted_count": int(drafted_per_row.sum().item()),
            "accepted_count": int(agreed_per_row.sum().item()),
        },
    }

    if step_idx == total_steps - 1 or not (next_gen == d_mask).any():
        result["results"] = decode_results(drafter, next_full_content, prompts, prompt_len)
        result["done"] = True

    return result