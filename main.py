import argparse
from src.registry import ModelRegistry
from src.model import ModelBroker
from src.runtime import run

def parse_dynamic_kwargs(unknown_args_list):
    """Converts a list of unknown CLI args into a **kwargs dictionary."""
    kwargs = {}
    i = 0
    while i < len(unknown_args_list):
        arg = unknown_args_list[i]
        if arg.startswith('--'):
            key = arg.lstrip('-')
            if i + 1 < len(unknown_args_list) and not unknown_args_list[i+1].startswith('--'):
                val_str = unknown_args_list[i+1]
                if val_str.isdigit(): val = int(val_str)
                else:
                    try: val = float(val_str)
                    except ValueError: val = val_str
                kwargs[key] = val
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            i += 1
    print(kwargs)
    return kwargs

def main():
    parser = argparse.ArgumentParser() # Removed fromfile_prefix_chars
    parser.add_argument("--drafter", type=str, required=True)
    parser.add_argument("--verifier", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    
    # Prompt inputs
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="Single line prompt string")
    group.add_argument("--prompt_file", type=str, help="Path to text or jsonl file with prompts")
    
    # Output file
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save results. If not provided, results are not saved.")
    
    parser.add_argument("--verbose", action="store_true", help="Print detailed step-by-step traces")
    
    known_args, unknown_args = parser.parse_known_args()
    custom_kwargs = parse_dynamic_kwargs(unknown_args)
    if known_args.verbose:
        custom_kwargs["verbose"] = True

    registry = ModelRegistry("registry.yaml")
    
    drafter_module, drafter_config = registry.load_model(known_args.drafter)
    drafter = ModelBroker(drafter_module, drafter_config, role="DRAFTER")

    verifier_module, verifier_config = registry.load_model(known_args.verifier)
    verifier = ModelBroker(verifier_module, verifier_config, role="VERIFIER")

    algorithm_func = registry.load_algorithm(known_args.algorithm)

    # Resolve prompts
    import os
    import json
    prompts = []
    if known_args.prompt:
        prompts.append(known_args.prompt)
    elif known_args.prompt_file and os.path.isfile(known_args.prompt_file):
        with open(known_args.prompt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Simple heuristic: if it looks like json dict, grab prompt key
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                        prompts.append(data.get("prompt", line))
                    except:
                        prompts.append(line)
                else:
                    prompts.append(line)

    print(f"\n[SYSTEM] Commencing Speculative Run...")
    print(f"         -> Drafter: {known_args.drafter}")
    print(f"         -> Verifier: {known_args.verifier}")
    print(f"         -> Algorithm: {known_args.algorithm}")
    print(f"         -> Custom Kwargs: {custom_kwargs}")
    print(f"[SYSTEM] Total prompts to process: {len(prompts)}\n")

    output_filepath = None
    if known_args.output_dir:
        os.makedirs(known_args.output_dir, exist_ok=True)
        output_filepath = os.path.join(known_args.output_dir, "output.jsonl")
        with open(output_filepath, 'w') as f:
            pass

    for i, p in enumerate(prompts):
        print(f"\n--- Processing Prompt {i+1}/{len(prompts)} ---")
        
        # We inject the current prompt into known_args seamlessly
        known_args.prompt = p
        
        output_data = run(drafter=drafter, 
            verifier=verifier, 
            algorithm_func=algorithm_func,
            known_args=known_args, 
            custom_kwargs=custom_kwargs)
            
        # Write to JSONL
        if output_filepath:
            with open(output_filepath, 'a') as f:
                json.dump({
                    "prompt": p,
                    "output": output_data["text"],
                    "metrics": output_data["metrics"]
                }, f)
                f.write('\n')
    
    # except Exception as e:
    #     print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()