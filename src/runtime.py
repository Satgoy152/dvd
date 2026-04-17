import torch
import time
import torch.nn.functional as F


def run(drafter, verifier, algorithm_func, prompts, custom_kwargs):
    """
    Run batched speculative diffusion generation.

    Args:
        drafter: ModelBroker for the drafter model.
        verifier: ModelBroker for the verifier model.
        algorithm_func: The step function from the selected algorithm.
        prompts: list[str] — one or more prompts to generate from.
        custom_kwargs: dict of runtime settings (steps, gen_length, mask_id,
                       batch_size, verbose, compute_divergence, etc.)

    Returns:
        dict with:
            "results": list[dict] — per-prompt {"text", "tokens", "prompt"}
            "aggregated_metrics": dict — metrics aggregated over all steps/batches
    """
    steps = int(custom_kwargs["steps"])
    gen_length = int(custom_kwargs["gen_length"])
    mask_id = int(custom_kwargs.get("mask_id", 126336))
    batch_size = int(custom_kwargs.get("batch_size", 1))

    device = verifier.model_state["model"].device
    tokenizer = verifier.model_state["tokenizer"]

    # Ensure a pad token is available (distinct from mask_id)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    all_results = []
    aggregated_metrics = {
        "drafter_nfe": 0,
        "verifier_nfe": 0,
        "total_latency": 0.0,
        "step_latency_sum": 0.0,
    }

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    verbose = custom_kwargs.get("verbose", False)

    print(f"\n[SYSTEM] Starting generation loop for {steps} steps "
          f"| {len(prompts)} prompt(s) in {num_batches} batch(es) of up to {batch_size}...\n")

    start_time_total = time.perf_counter()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        B = len(batch_prompts)

        if verbose or num_batches > 1:
            print(f"\n[SYSTEM] === Batch {batch_idx + 1}/{num_batches} ({B} prompt(s)) ===")

        # ----- Tokenize each prompt via the adapter -----
        token_seqs = []
        prompt_lengths = []
        for p in batch_prompts:
            toks = drafter.tokenize(p, **custom_kwargs)["input_ids"].squeeze(0)  # (L_i,)
            token_seqs.append(toks)
            prompt_lengths.append(toks.shape[0])

        max_prompt_len = max(prompt_lengths)
        seq_len = max_prompt_len + gen_length

        if verbose:
            print(f"[SYSTEM] Prompt lengths: {prompt_lengths} | "
                  f"max_prompt_len={max_prompt_len} | seq_len={seq_len}")

        # ----- Build batched tensor: [padded_prompt | MASK * gen_length] -----
        # Pad shorter prompts with pad_id (NOT mask_id) so sample_verifier
        # doesn't treat padding positions as generation candidates.
        current_tokens = torch.full(
            (B, seq_len), mask_id, dtype=torch.long, device=device
        )
        attention_mask = torch.ones((B, seq_len), dtype=torch.long, device=device)

        for i, (seq, plen) in enumerate(zip(token_seqs, prompt_lengths)):
            current_tokens[i, :plen] = seq.to(device)
            # Fill the gap between prompt end and max_prompt_len with pad_id
            if plen < max_prompt_len:
                current_tokens[i, plen:max_prompt_len] = pad_id
                attention_mask[i, plen:max_prompt_len] = 0

        # Add attention mask to kwargs so it flows to drafter/verifier via algorithms
        custom_kwargs["attention_mask"] = attention_mask

        # ----- Generation loop -----
        for s in range(steps):
            step_kwargs = dict(custom_kwargs)
            start_time_step = time.perf_counter()

            output = algorithm_func(
                current_tokens=current_tokens,
                drafter=drafter,
                verifier=verifier,
                step_idx=s,
                total_steps=steps,
                **step_kwargs,
            )

            current_tokens = output.get("next_tokens", current_tokens)

            step_latency = time.perf_counter() - start_time_step
            aggregated_metrics["step_latency_sum"] += step_latency

            metrics = output.get("metrics", {})
            if "step_latency" not in metrics:
                metrics["step_latency"] = step_latency

            for k, v in metrics.items():
                if k in aggregated_metrics:
                    aggregated_metrics[k] += v
                else:
                    aggregated_metrics[k] = v

            if verbose:
                print(f"[SYSTEM] Step {s + 1}/{steps} done. Metrics this step: {metrics}")

        # ----- Divergence check (optional) -----
        if custom_kwargs.get("compute_divergence", True):
            with torch.no_grad():
                div_kwargs = {k: v for k, v in custom_kwargs.items() if k != "steps"}
                ver_out = verifier.generate(
                    input_toks={"input_ids": current_tokens}, steps=steps, **div_kwargs
                )
                logits = ver_out["logits"]
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # (B, seq_len)
                # Compute entropy only over the generation region
                avg_entropy = entropy[:, max_prompt_len:].mean().item()
                aggregated_metrics["divergence_entropy"] = (
                    aggregated_metrics.get("divergence_entropy", 0.0) + avg_entropy
                )

        # ----- Decode per-prompt results -----
        for i in range(B):
            # Decode only the generation region (after max_prompt_len)
            gen_tokens = current_tokens[i, max_prompt_len:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            all_results.append({
                "text": text,
                "tokens": current_tokens[i: i + 1].clone(),
                "prompt": batch_prompts[i],
            })

            if verbose:
                print(f"\n--- Result {len(all_results)} ---")
                print(f"  Prompt: {batch_prompts[i][:80]}...")
                print(f"  Output: {text[:120]}...")

    # ----- Finalize global metrics -----
    total_time = time.perf_counter() - start_time_total
    aggregated_metrics["total_latency"] = total_time
    total_step_count = steps * num_batches
    aggregated_metrics["avg_step_latency"] = (
        aggregated_metrics["step_latency_sum"] / total_step_count
        if total_step_count > 0 else 0.0
    )

    # Print summary
    print("\n=== GENERATION COMPLETE ===")
    for r in all_results:
        print(f"\n--- Prompt: {r['prompt'][:100]} ---")
        print(f"    Output: {r['text'][:200]}")
    print(f"\nAggregated Metrics: {aggregated_metrics}")

    return {
        "results": all_results,
        "aggregated_metrics": aggregated_metrics,
    }
