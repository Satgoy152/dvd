#!/bin/bash
# Benchmark suite for Drafter Verifier Diffusion (DVD)

TASKS="gsm8k"
LIMIT=10 # For speed of benchmarking script. Adjust for real evaluations.
GEN_LENGTH=64
NUM_FEWSHOT=5
STEPS_ARR=(10 20 30)

VERIFIER="llada_8b_instruct"
DRAFTER="llada_8b_base"

echo "=== Starting DVD Evaluation Pipeline ==="

# 1. Baseline standard model generation (Algorithm: baseline_cascade)
for steps in "${STEPS_ARR[@]}"; do
    echo "Running Baseline Cascade ($steps steps)..."
    python scripts/run_evals.py \
      --drafter "$DRAFTER" \
      --verifier "$VERIFIER" \
      --algorithm "baseline_cascade" \
      --tasks "$TASKS" \
      --limit "$LIMIT" \
      --steps "$steps" \
      --gen_length "$GEN_LENGTH" \
      --num_fewshot "$NUM_FEWSHOT" \
      --output_file "baseline_${steps}steps.json"
done

# 2. Speculative Decode Threshold Algorithm
for steps in "${STEPS_ARR[@]}"; do
    echo "Running Threshold Verifier ($steps steps)..."
    python scripts/run_evals.py \
      --drafter "$DRAFTER" \
      --verifier "$VERIFIER" \
      --algorithm "threshold_verifier" \
      --tasks "$TASKS" \
      --limit "$LIMIT" \
      --steps "$steps" \
      --gen_length "$GEN_LENGTH" \
      --num_fewshot "$NUM_FEWSHOT" \
      --output_file "threshold_${steps}steps.json"
done

# 3. Speculative Decode Top-K Algorithm
for steps in "${STEPS_ARR[@]}"; do
    echo "Running Top-K Intersection ($steps steps)..."
    python scripts/run_evals.py \
      --drafter "$DRAFTER" \
      --verifier "$VERIFIER" \
      --algorithm "top_k_intersection" \
      --tasks "$TASKS" \
      --limit "$LIMIT" \
      --steps "$steps" \
      --gen_length "$GEN_LENGTH" \
      --num_fewshot "$NUM_FEWSHOT" \
      --output_file "topk_${steps}steps.json"
done

echo "=== All evaluations complete ==="
echo "Generating Pareto Frontier plots..."

python scripts/plot_metrics.py

echo "Plots saved in eval_results/plots/"
