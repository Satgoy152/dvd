#!/bin/bash

# Ensure the script stops if there's an error
set -e

export HF_HOME="/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/huggingface"

echo "Starting DVD Speculative Baseline Run..."

python main.py \
  --drafter "llada_8b_base" \
  --verifier "llada_8b_instruct" \
  --algorithm "baseline_cascade" \
  --prompt "Give me a short story for a 5 year old" \
  --output_dir "baseline_output" \
  --tokenizer_max_len 32 \
  --gen_length 64 \
  --steps 32
