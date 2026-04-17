#!/bin/bash
# Mini Benchmark Script for DVD
# Use this for quick testing and debugging!

# --- Configuration variables you can change ---
ALGORITHM="top_k_intersection"  # try: baseline_cascade, threshold_verifier, top_k_intersection
VERIFIER="llada_8b_instruct"
DRAFTER="llada_8b_base"

TASK="gsm8k"
LIMIT=2            # Evaluates on just 2 examples (very fast)
STEPS=5            # Small step limit
GEN_LENGTH=16      # Short generation length
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
  --output_file "mini_benchmark_out.json"

echo "Mini benchmark complete! Check eval_results/mini_benchmark_out.json for metrics."

python scripts/plot_metrics.py
