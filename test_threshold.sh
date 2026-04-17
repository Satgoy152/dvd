#!/bin/bash
# A quick end-to-end integration test to verify the threshold algorithm loop logic

# Test 1: Strict Threshold (0.99) - Should reject almost all drafter tokens and fallback to normal model
echo "========== TESTING STRICT THRESHOLD (0.99) =========="
python main.py \
  --drafter "llada_8b_instruct" \
  --verifier "llada_8b_instruct" \
  --algorithm "threshold_verifier" \
  --prompt "What is the capital of France?" \
  --output_dir "test_output" \
  --tokenizer_max_len 32 \
  --gen_length 32 \
  --steps 16 \
  --threshold 0.99 \
  --verbose

echo "\n\n========== TESTING LENIENT THRESHOLD (0.01) =========="
# Test 2: Lenient Threshold (0.01) - Should accept almost every drafter token
python main.py \
  --drafter "llada_8b_instruct" \
  --verifier "llada_8b_instruct" \
  --algorithm "threshold_verifier" \
  --prompt "What is the capital of France?" \
  --output_dir "test_output" \
  --tokenizer_max_len 32 \
  --gen_length 32 \
  --steps 16 \
  --threshold 0.01 \
  --verbose

echo "\nCheck the printed 'Aggregated Metrics' above. 
For 0.99 threshold, accepted_count should be extremely low. 
For 0.01 threshold, accepted_count should be near the drafted_count."
