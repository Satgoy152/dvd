#!/bin/bash
# Mini Benchmark Script for DVD
# Use this for quick testing and debugging!
set -e

export HF_HOME="/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/huggingface"

# --- Configuration variables you can change ---
ALGORITHM="baseline_cascade"  # try: baseline_cascade, threshold_verifier, top_k_intersection
VERIFIER="llada_8b_instruct"
DRAFTER="llada_8b_instruct"

TASK="gsm8k"
LIMIT=24           
STEPS=64            # Small step limit
GEN_LENGTH=512      # Short generation length
NUM_FEWSHOT=0      # Few-shot examples
BATCH_SIZE=1        # Number of samples per batch
# ----------------------------------------------

echo "Running rapid test for algorithm: $ALGORITHM"
echo "Evaluating $LIMIT samples on $TASK..."

python scripts/run_evals.py \
  --drafter "$DRAFTER" \
  --verifier "$VERIFIER" \
  --algorithm "$ALGORITHM" \
  --tasks "$TASK" \
  --limit "$LIMIT" \
  --steps "$STEPS" \
  --gen_length "$GEN_LENGTH" \
  --num_fewshot "$NUM_FEWSHOT" \
  --batch_size "$BATCH_SIZE" \
  --output_file "mini_benchmark_out.json"

echo "Mini benchmark complete! Check eval_results/mini_benchmark_out.json for metrics."

python scripts/plot_metrics.py
