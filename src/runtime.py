import torch


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


def run(drafter, verifier, known_args, custom_kwargs):
    steps = int(custom_kwargs["steps"])
    gen_length = int(custom_kwargs["gen_length"])
    mask_id = int(custom_kwargs.get("mask_id", 126336))

    # Tokenize the prompt
    drafter_tokens = drafter.tokenize(known_args.prompt, **custom_kwargs)
    prompt_ids = drafter_tokens["input_ids"]
    prompt_len = prompt_ids.shape[1]
    device = verifier.model_state["model"].device
    print(f"\n[SYSTEM] Prompt tokenized to len: {prompt_len}\n")

    # Initialize: prompt + [MASK] * gen_length (on model device)
    current_tokens = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device
    )
    current_tokens[:, :prompt_len] = prompt_ids.to(device)

    for s in range(steps):
        step_kwargs = dict(custom_kwargs, steps=s + 1)  # non-zero → adapter uses input as-is

        # Step C: Verifier runs a forward pass on the current sequence
        verifier_output = verifier.generate(
            input_toks={"input_ids": current_tokens},
            **step_kwargs
        )
        print(f"\n[SYSTEM] Verifier step {s+1}/{steps} done\n")

        # Step D: Unmask the top-K most confident masked tokens
        current_tokens = sample_verifier(
            verifier_output["logits"],
            current_tokens,
            mask_id,
            gen_length,
            steps,
        )

    # Decode final output (only the generated portion after the prompt)
    tokenizer = verifier.model_state["tokenizer"]
    final_text = tokenizer.decode(current_tokens[0][prompt_len:], skip_special_tokens=True)

    print("\n=== FINAL VERIFIED OUTPUT ===")
    print(f"Prompt: {known_args.prompt}")
    print(f"Output: {final_text}")
