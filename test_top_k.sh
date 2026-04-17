#!/bin/bash
# A quick end-to-end integration test to verify the top_k_intersection algorithm loop logic

echo "========== TESTING TOP-K INTERSECTION VERIFIER =========="
python main.py \
  --drafter "llada_8b_instruct" \
  --verifier "llada_8b_instruct" \
  --algorithm "top_k_intersection" \
  --prompt "What is the capital of France?" \
  --output_dir "test_output" \
  --tokenizer_max_len 32 \
  --gen_length 32 \
  --steps 16 \
  --verbose

echo "\nCheck the printed 'Aggregated Metrics' above. 
If both models are the identical checkpoint, accepted_count should ideally perfectly match drafted_count since both models will always argmax to the same exact top-confidence tokens."
