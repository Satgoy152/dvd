import torch

from algorithms._common import (
    initialize_content,
    step_generate,
    sample_topk_confident,
    decode_results,
)


def run_step(drafter, verifier, prompts, step_idx, total_steps, algo_state, **kwargs):
    """Baseline diffusion: each step, unmask the top-K most confident masked
    positions according to the verifier's logits. No drafter involvement.
    """
    gen_length = int(kwargs.get("gen_length"))
    mask_id = int(kwargs.get("mask_id", 126336))

    if gen_length % total_steps != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be divisible by total_steps ({total_steps})"
        )
    tokens_per_step = gen_length // total_steps

    kwargs_pass = {k: v for k, v in kwargs.items() if k not in ["gen_length", "mask_id"]}
    if step_idx == 0:
        full_content, full_attn, prompt_len, logits = initialize_content(
            verifier, prompts, gen_length, mask_id, **kwargs_pass
        )
    else:
        full_content = algo_state["content"]
        full_attn = algo_state["content_attn"]
        prompt_len = algo_state["prompt_len"]
        logits = step_generate(
            verifier, full_content, full_attn, prompt_len, gen_length, mask_id
        )

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

    if step_idx == total_steps - 1:
        result["results"] = decode_results(verifier, full_content, prompts, prompt_len)

    return result
