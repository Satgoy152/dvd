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

def sample_verifier(logits, tokens, mask_id, gen_length, steps):
    """Unmask the top gen_length//steps most confident masked tokens each step.

    Args:
        logits: (batch, seq_len, vocab_size) model output logits.
        tokens: (batch, seq_len) current token ids (some are mask_id).
        mask_id: the id used for masked positions.
        gen_length: total number of tokens to generate.
        steps: total number of diffusion steps.

    Returns:
        Updated tokens tensor with the top-K masked positions replaced by
        the model's most confident predictions.
    """
    if gen_length % steps != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be evenly divisible by steps ({steps})"
        )

    tokens_per_step = gen_length // steps

    # Confidence = max logit value at each position
    confidence = logits.max(dim=-1).values          # (batch, seq_len)
    predicted_ids = logits.argmax(dim=-1)            # (batch, seq_len)

    # Only consider positions that are currently masked
    mask_positions = tokens == mask_id               # (batch, seq_len)

    # Zero out confidence for non-masked positions so they are never selected
    confidence = confidence.masked_fill(~mask_positions, float('-inf'))

    # Select the top-K most confident masked positions
    _, topk_indices = confidence.topk(tokens_per_step, dim=-1)  # (batch, K)

    # Unmask those positions with the predicted token ids
    updated_tokens = tokens.clone()
    updated_tokens.scatter_(1, topk_indices, predicted_ids.gather(1, topk_indices))

    return updated_tokens
