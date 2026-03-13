#!/bin/bash

# Ensure the script stops if there's an error
set -e

export HF_HOME="/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/huggingface"

echo "Starting LLaDA Speculative Run..."

python main.py \
  --drafter "llada_8b_instruct" \
  --verifier "llada_8b_instruct" \
  --prompt "Give me a short story for a 5 year old" \
  --tokenizer_max_len 32 \
  --gen_length 64 \
  --steps 128