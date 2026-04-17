import time


def run(drafter, verifier, algorithm_func, prompts, custom_kwargs):
    """Model-agnostic orchestration: loops outer steps, aggregates metrics,
    and hands everything to the algorithm. The algorithm owns tokenization,
    state, sampling, and decoding.

    Algorithm contract:
        algorithm_func(
            drafter, verifier, prompts, step_idx, total_steps,
            algo_state, **custom_kwargs,
        ) -> {
            "algo_state": dict    # opaque; threaded into next step
            "metrics":    dict    # aggregated by runtime
            "results":    list    # optional, returned on final step
            "done":       bool    # optional, early-exit signal
        }
    """
    steps = int(custom_kwargs["steps"])
    batch_size = int(custom_kwargs.get("batch_size", 1))
    verbose = custom_kwargs.get("verbose", False)

    aggregated_metrics = {
        "drafter_nfe": 0,
        "verifier_nfe": 0,
        "total_latency": 0.0,
        "step_latency_sum": 0.0,
    }
    all_results = []

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    print(f"\n[SYSTEM] Starting generation: {steps} steps | "
          f"{len(prompts)} prompt(s) in {num_batches} batch(es) of up to {batch_size}...\n")

    start_time_total = time.perf_counter()
    total_step_count = 0

    for batch_idx in range(num_batches):
        batch_prompts = prompts[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        if verbose or num_batches > 1:
            print(f"\n[SYSTEM] === Batch {batch_idx + 1}/{num_batches} "
                  f"({len(batch_prompts)} prompt(s)) ===")

        algo_state = {}
        output = None

        for s in range(steps):
            start_time_step = time.perf_counter()
            output = algorithm_func(
                drafter=drafter,
                verifier=verifier,
                prompts=batch_prompts,
                step_idx=s,
                total_steps=steps,
                algo_state=algo_state,
                **custom_kwargs,
            )
            step_latency = time.perf_counter() - start_time_step
            aggregated_metrics["step_latency_sum"] += step_latency
            total_step_count += 1

            algo_state = output.get("algo_state", algo_state)

            metrics = output.get("metrics", {})
            metrics.setdefault("step_latency", step_latency)
            for k, v in metrics.items():
                if k in aggregated_metrics:
                    aggregated_metrics[k] += v
                else:
                    aggregated_metrics[k] = v

            if verbose:
                print(f"[SYSTEM] Step {s + 1}/{steps} metrics: {metrics}")

            if output.get("done", False):
                break

        if output is not None:
            all_results.extend(output.get("results", []))

    aggregated_metrics["total_latency"] = time.perf_counter() - start_time_total
    aggregated_metrics["avg_step_latency"] = (
        aggregated_metrics["step_latency_sum"] / total_step_count
        if total_step_count > 0 else 0.0
    )

    print("\n=== GENERATION COMPLETE ===")
    for r in all_results:
        print(f"\n--- Prompt: {r['prompt'][:100]} ---")
        print(f"    Output: {r['text'][:200]}")
    print(f"\nAggregated Metrics: {aggregated_metrics}")

    return {"results": all_results, "aggregated_metrics": aggregated_metrics}
