#!/bin/bash

# Ensure the script stops if there's an error
set -e


echo "Starting DVD Speculative Baseline Run..."

python main.py \
  --drafter "mdlm_instruct" \
  --verifier "mdlm_instruct" \
  --algorithm "baseline_cascade" \
  --prompt "Give me a short story for a 5 year old" \
  --output_dir "baseline_output" \
  --tokenizer_max_len 256 \
  --gen_length 256 \
  --steps 32
