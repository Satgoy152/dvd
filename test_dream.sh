#!/bin/bash

# Ensure the script stops if there's an error
set -e

# export HF_HOME="/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/huggingface"

echo "Starting DVD Speculative Baseline Run..."

python main.py \
  --drafter "dream_7b" \
  --verifier "dream_7b" \
  --algorithm "baseline_cascade_dream" \
  --prompt "Give me a short story for a 5 year old" \
  --output_dir "baseline_output" \
  --tokenizer_max_len 256 \
  --gen_length 256 \
  --verbose \
  --steps 32
