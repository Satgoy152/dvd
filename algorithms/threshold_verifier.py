import torch
import torch.nn.functional as F

from algorithms._common import (
    initialize_content,
    step_generate,
    sample_topk_confident,
    decode_results,
)


def run_step(drafter, verifier, prompts, step_idx, total_steps, algo_state, **kwargs):
    """Threshold verifier:
    1. Drafter proposes top-K confident unmasks.
    2. Verifier scores the drafted tokens; keeps those with prob >= threshold.
    3. Rejected quota is refilled from verifier's own top-confidence masked positions.

    Assumes drafter and verifier share a tokenizer (content is passed between
    them verbatim). For cross-tokenizer setups, convert via src.token_utils.
    """
    gen_length = int(kwargs.get("gen_length"))
    mask_id = int(kwargs.get("mask_id", 126336))
    threshold = float(kwargs.get("threshold", 0.5))

    if gen_length % total_steps != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be divisible by total_steps ({total_steps})"
        )
    tokens_per_step = gen_length // total_steps

    # 1. Drafter forward on current state.
    if step_idx == 0:
        full_content, full_attn, prompt_len, drafter_logits = initialize_content(
            drafter, prompts, gen_length, mask_id, **kwargs
        )
    else:
        full_content = algo_state["content"]
        full_attn = algo_state["content_attn"]
        prompt_len = algo_state["prompt_len"]
        drafter_logits = step_generate(
            drafter, full_content, full_attn, prompt_len, gen_length, mask_id
        )

    gen_region = full_content[:, prompt_len:]

    # 2. Drafter proposes unmasks.
    drafted_gen = sample_topk_confident(drafter_logits, gen_region, mask_id, tokens_per_step)
    drafted_positions = (gen_region == mask_id) & (drafted_gen != mask_id)

    # 3. Verifier forward on the drafter's proposed content.
    drafted_content = torch.cat([full_content[:, :prompt_len], drafted_gen], dim=1)
    ver_logits = step_generate(
        verifier, drafted_content, full_attn, prompt_len, gen_length, mask_id
    )

    # 4. Accept drafted tokens whose verifier probability clears the threshold.
    ver_probs = F.softmax(ver_logits, dim=-1)
    drafted_probs = ver_probs.gather(2, drafted_gen.unsqueeze(-1)).squeeze(-1)
    passed_positions = drafted_positions & (drafted_probs >= threshold)

    next_gen = gen_region.clone()
    next_gen[passed_positions] = drafted_gen[passed_positions]

    # 5. Refill each row's rejected quota from its own verifier top-confidence picks.
    drafted_per_row = drafted_positions.sum(dim=-1)
    passed_per_row = passed_positions.sum(dim=-1)
    failed_per_row = drafted_per_row - passed_per_row
    max_failed = int(failed_per_row.max().item())
    if max_failed > 0:
        remaining_mask = next_gen == mask_id
        ver_confidence = ver_logits.max(dim=-1).values
        ver_predicted_ids = ver_logits.argmax(dim=-1)
        ver_confidence = ver_confidence.masked_fill(~remaining_mask, float("-inf"))
        _, topk_indices = ver_confidence.topk(max_failed, dim=-1)
        col_idx = torch.arange(max_failed, device=topk_indices.device).unsqueeze(0)
        valid = col_idx < failed_per_row.unsqueeze(1)
        new_vals = ver_predicted_ids.gather(1, topk_indices)
        orig_vals = next_gen.gather(1, topk_indices)
        scatter_src = torch.where(valid, new_vals, orig_vals)
        next_gen.scatter_(1, topk_indices, scatter_src)

    next_full_content = torch.cat([full_content[:, :prompt_len], next_gen], dim=1)

    drafted_count = int(drafted_per_row.sum().item())
    accepted_count = int(passed_per_row.sum().item())
    result = {
        "algo_state": {
            "content": next_full_content,
            "content_attn": full_attn,
            "prompt_len": prompt_len,
        },
        "metrics": {
            "drafter_nfe": 1,
            "verifier_nfe": 1,
            "drafted_count": drafted_count,
            "accepted_count": accepted_count,
            "acceptance_rate": accepted_count / max(1, drafted_count),
        },
    }

    if step_idx == total_steps - 1:
        result["results"] = decode_results(verifier, next_full_content, prompts, prompt_len)

    return result
