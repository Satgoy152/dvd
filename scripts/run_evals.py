import argparse
import json
import os
import time
from pathlib import Path
from lm_eval import simple_evaluate

# Add src to python path for internal imports to work
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evals.dvd_lm_eval import DVD_LM

def main():
    parser = argparse.ArgumentParser(description="Evaluate DVD using lm-evaluation-harness")
    parser.add_argument("--drafter", type=str, required=True, help="Registry name of the drafter model")
    parser.add_argument("--verifier", type=str, required=True, help="Registry name of the verifier model")
    parser.add_argument("--algorithm", type=str, required=True, help="Registry name of the verification algorithm")
    parser.add_argument("--tasks", type=str, default="gsm8k", help="Comma-separated list of tasks, e.g. gsm8k,hellaswag")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples per task to evaluate")
    parser.add_argument("--steps", type=int, default=10, help="Number of generation steps")
    parser.add_argument("--gen_length", type=int, default=64, help="Number of tokens to generate")
    parser.add_argument("--swap_models_each_step", action="store_true", help="Offload to CPU aggressively to simulate single-GPU constraints")
    parser.add_argument("--output_file", type=str, default=None, help="Custom output JSON name")
    args = parser.parse_args()

    print(f"[{args.algorithm}] Starting evaluation on tasks: {args.tasks}")
    
    # Initialize the DVD_LM Wrapper
    model = DVD_LM(
        drafter_name=args.drafter,
        verifier_name=args.verifier,
        algorithm_name=args.algorithm,
        steps=args.steps,
        gen_length=args.gen_length,
        swap_models_each_step=args.swap_models_each_step,
        registry_path=os.path.join(os.path.dirname(__file__), "..", "registry.yaml")
    )

    t0 = time.time()
    
    # Run evaluation
    results = simple_evaluate(
        model=model,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        batch_size=1, # DVD generates sequences currently best tested with BS=1
        limit=args.limit
    )
    
    t1 = time.time()
    eval_time = t1 - t0
    
    # Process custom metrics from the Model wrapper
    metrics = model.total_aggregated_metrics
    req_count = metrics.get("request_count", 1)
    
    avg_metrics = {k: v / req_count for k, v in metrics.items() if k != "request_count"}
    
    print("\n=== Eval Finished ===")
    print(f"Total Eval Time: {eval_time:.2f}s")
    print(f"Average Metrics over {req_count} requests: {avg_metrics}")
    
    # Extract task accuracies (generation eval)
    # lm-eval uses varied key formats like "exact_match,flexible-extract", "acc,none", etc.
    task_scores = {}
    for task_name, task_metrics in results.get("results", {}).items():
        score = None
        # Priority order: flexible exact_match > strict exact_match > acc
        for key in task_metrics:
            if "exact_match" in key and "stderr" not in key and "flexible" in key:
                score = task_metrics[key]
                break
        if score is None:
            for key in task_metrics:
                if "exact_match" in key and "stderr" not in key:
                    score = task_metrics[key]
                    break
        if score is None:
            for key in task_metrics:
                if key.startswith("acc") and "stderr" not in key:
                    score = task_metrics[key]
                    break
        task_scores[task_name] = score
        
    print(f"Task Scores: {task_scores}")
    
    # Package output payload
    output_data = {
        "config": vars(args),
        "results": results["results"],
        "dvd_metrics_average": avg_metrics,
        "task_scores": task_scores,
        "eval_time": eval_time
    }
    
    # Dump to json
    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)
    
    out_name = args.output_file if args.output_file else f"{args.algorithm}_{args.verifier}.json"
    output_path = output_dir / out_name
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
