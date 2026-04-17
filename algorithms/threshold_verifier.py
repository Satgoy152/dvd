import torch
import torch.nn.functional as F
from src.utils import sample_verifier

def run_step(current_tokens, drafter, verifier, step_idx, total_steps, **kwargs):
    """
    Threshold Verifier Algorithm:
    1. Drafter takes a step to propose unmasking K tokens.
    2. Verifier takes a forward pass on the Drafter's proposed sequence.
    3. We check the Verifier's confidence for the tokens the drafter chose.
       - If confidence >= threshold, the token is accepted.
       - If confidence < threshold, the token is rejected.
    4. To maintain the generation schedule, if M tokens are rejected, the Verifier 
       re-samples M top-confident tokens from the remaining mask pool.
    """
    gen_length = int(kwargs.get("gen_length"))
    mask_id = int(kwargs.get("mask_id", 126336))
    threshold = float(kwargs.get("threshold", 0.5))
    
    # 1. Run Drafter
    drafter_out = drafter.generate(input_toks={"input_ids": current_tokens}, steps=step_idx+1, **kwargs)
    
    # Simulate Drafter unmasking tokens
    drafted_tokens = sample_verifier(
        logits=drafter_out["logits"], 
        tokens=current_tokens, 
        mask_id=mask_id, 
        gen_length=gen_length, 
        steps=total_steps
    )
    
    # 2. Run Verifier on the Drafter's output
    ver_out = verifier.generate(input_toks={"input_ids": drafted_tokens}, steps=step_idx+1, **kwargs)
    ver_logits = ver_out["logits"]
    
    # 3. Verification
    ver_probs = F.softmax(ver_logits, dim=-1)
    
    # Get the verifier's probability for the specific tokens the drafter chose
    drafted_probs = ver_probs.gather(2, drafted_tokens.unsqueeze(-1)).squeeze(-1) # (batch, seq_len)
    
    # Positions drafter unmasked
    drafted_positions = (current_tokens == mask_id) & (drafted_tokens != mask_id)
    
    # Positions that pass the threshold
    passed_positions = drafted_positions & (drafted_probs >= threshold)
    
    failed_count = drafted_positions.sum().item() - passed_positions.sum().item()
    
    next_tokens = current_tokens.clone()
    next_tokens[passed_positions] = drafted_tokens[passed_positions]
    
    # 4. Fallback (Fill the quota of rejected tokens using Verifier's own top predictions)
    if failed_count > 0:
        remaining_mask = next_tokens == mask_id
        
        ver_confidence = ver_logits.max(dim=-1).values
        ver_predicted_ids = ver_logits.argmax(dim=-1)
        
        # We only consider tokens that are still masked
        ver_confidence = ver_confidence.masked_fill(~remaining_mask, float('-inf'))
        
        _, topk_indices = ver_confidence.topk(failed_count, dim=-1)
        next_tokens.scatter_(1, topk_indices, ver_predicted_ids.gather(1, topk_indices))
        
    return {
        "next_tokens": next_tokens,
        "metrics": {
            "drafter_nfe": 1,
            "verifier_nfe": 1,
            "drafted_count": drafted_positions.sum().item(),
            "accepted_count": passed_positions.sum().item(),
            "acceptance_rate": passed_positions.sum().item() / max(1, drafted_positions.sum().item())
        }
    }
