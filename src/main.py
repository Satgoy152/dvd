import argparse
from registry import ModelRegistry
from model import ModelBroker

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
    parser.add_argument("--prompt", type=str, required=True)
    
    known_args, unknown_args = parser.parse_known_args()
    custom_kwargs = parse_dynamic_kwargs(unknown_args)

    try:
        registry = ModelRegistry("registry.yaml")
        
        drafter_module, drafter_config = registry.load_model(known_args.drafter)
        drafter = ModelBroker(drafter_module, drafter_config, role="DRAFTER")

        verifier_module, verifier_config = registry.load_model(known_args.verifier)
        verifier = ModelBroker(verifier_module, verifier_config, role="VERIFIER")

        print(f"\n[SYSTEM] Commencing Speculative Run with custom kwargs: {custom_kwargs}\n")

        # Step A: Drafter Tokenizes (now receives filtered custom_kwargs)
        drafter_tokens = drafter.tokenize(known_args.prompt, **custom_kwargs)

        print(f"\n[SYSTEM] Drafter prompt tokenized of len: {len(drafter_tokens)}\n")
        
        # Step B: Drafter Generates
        drafter_output = drafter.generate(drafter_tokens, **custom_kwargs)

        print(f"\n[SYSTEM] Drafter generate done\n")
        
        # Step C: Verifier Tokenizes
        verifier_tokens = verifier.tokenize(known_args.prompt, **custom_kwargs)

        print(f"\n[SYSTEM] Verifer prompt tokenized of len {len(verifier_tokens)}\n")
        
        # Step D: Verifier Generates (takes the drafted latents)
        final_latents = verifier.generate(
            input_toks=verifier_tokens, 
            **custom_kwargs
        )

        print("\n=== FINAL VERIFIED OUTPUT ===")
        print(final_latents)

    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()