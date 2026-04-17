import torch

def run(drafter, verifier, algorithm_func, known_args, custom_kwargs):
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

    # Initialize aggregate metrics
    aggregated_metrics = {
        "drafter_nfe": 0,
        "verifier_nfe": 0
    }

    print(f"\n[SYSTEM] Starting generation loop for {steps} steps...\n")

    for s in range(steps):
        step_kwargs = dict(custom_kwargs)
        
        # Call the user-provided algorithm
        output = algorithm_func(
            current_tokens=current_tokens,
            drafter=drafter,
            verifier=verifier,
            step_idx=s,
            total_steps=steps,
            **step_kwargs
        )
        
        current_tokens = output.get("next_tokens", current_tokens)
        metrics = output.get("metrics", {})
        
        for k, v in metrics.items():
            if k in aggregated_metrics:
                aggregated_metrics[k] += v
            else:
                aggregated_metrics[k] = v

        if custom_kwargs.get("verbose"):
            print(f"[SYSTEM] Step {s+1}/{steps} done. Metrics this step: {metrics}")

    # Decode final output (only the generated portion after the prompt)
    tokenizer = verifier.model_state["tokenizer"]
    final_text = tokenizer.decode(current_tokens[0][prompt_len:], skip_special_tokens=True)

    print("\n=== FINAL VERIFIED OUTPUT ===")
    print(f"Prompt: {known_args.prompt}")
    print(f"Output: {final_text}")
    print(f"Aggregated Metrics: {aggregated_metrics}")
    
    return {
        "text": final_text,
        "tokens": current_tokens,
        "metrics": aggregated_metrics
    }
