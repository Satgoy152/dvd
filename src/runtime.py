import torch
import time
import torch.nn.functional as F

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
        "verifier_nfe": 0,
        "total_latency": 0.0,
        "step_latency_sum": 0.0,
    }

    print(f"\n[SYSTEM] Starting generation loop for {steps} steps...\n")

    start_time_total = time.perf_counter()

    for s in range(steps):
        step_kwargs = dict(custom_kwargs)
        start_time_step = time.perf_counter()
        
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

        if custom_kwargs.get("verbose"):
            print(f"[SYSTEM] Step {s+1}/{steps} done. Metrics this step: {metrics}")

    total_time = time.perf_counter() - start_time_total
    aggregated_metrics["total_latency"] = total_time
    aggregated_metrics["avg_step_latency"] = aggregated_metrics["step_latency_sum"] / steps if steps > 0 else 0

    # Divergence Check
    if custom_kwargs.get("compute_divergence", True):
        with torch.no_grad():
            ver_out = verifier.generate(input_toks={"input_ids": current_tokens}, steps=steps, **custom_kwargs)
            logits = ver_out["logits"]
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) # (1, seq_len)
            avg_entropy = entropy[:, prompt_len:].mean().item()
            aggregated_metrics["divergence_entropy"] = avg_entropy

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
