# DVD
We propose DrafterVerifierDiffusion (DVD), a speculative decoding framework that combines both approaches: a lightweight KV-cached MDLM drafts multiple denoising steps rapidly, while a bidirectional MDLM verifies outputs. *WIP*

## Setup
Install the necessary requirements via `uv` or any standard Python package manager based on the `pyproject.toml`.

## Project Structure
The project is divided into core logic and user-provided interfaces:
- `src/`: Contains the core math and background engine logic (`runtime.py`, `model.py`, `utils.py`, `evals`).
- `adapters/`: Contains the user-provided Python scripts defining the interface for initializing and calling various diffusion models.
- `algorithms/`: Contains the user-provided Python scripts defining the verification schedules and data flow logic.
- `registry.yaml`: The configuration hub mapping human-readable names to your custom models and algorithm scripts.
- `main.py`: The CLI entrypoint for executing runs.

## Inference Usage
Run speculative generation via `main.py`. 

```bash
python main.py \
  --drafter "llada_8b_base" \
  --verifier "llada_8b_instruct" \
  --algorithm "baseline_cascade" \
  --prompt "Give me a short story for a 5 year old" \
  --output_dir "baseline_output" \
  --tokenizer_max_len 32 \
  --gen_length 64 \
  --steps 32
```

### Command Line Flags
- `--drafter`: Name of the drafter model (from `registry.yaml`).
- `--verifier`: Name of the verifier model (from `registry.yaml`).
- `--algorithm`: Name of the verification algorithm (from `registry.yaml`).
- `--prompt`: A single prompt string. *(Mutually exclusive with `--prompt_file`)*
- `--prompt_file`: Path to a text or JSONL file containing a list of prompts.
- `--output_dir`: (Optional) Directory to save the final `output.jsonl` results.
- `--verbose`: (Optional) Prints detailed trace logs for adapters and verification steps.
- `**kwargs`: Any extra arguments (e.g., `--steps`, `--gen_length`) are dynamically bundled and passed directly into your loaded algorithm and adapter scripts.
