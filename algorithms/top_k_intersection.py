import torch
import torch.nn.functional as F
from src.utils import sample_verifier

def run_step(current_tokens, drafter, verifier, step_idx, total_steps, **kwargs):
    """
    Top-K Intersection Verifier Algorithm:
    1. Drafter takes a forward pass on the current state and proposes tokens to unmask.
    2. Verifier takes a forward pass on the Drafter's drafted tokens.
    3. We check if the exact tokens the drafter chose at the modified positions 
       match the Verifier's highest confidence predicted tokens for those positions.
    4. To maintain the generation schedule, wherever they disagree, the Verifier 
       overrides the Drafter and fills the unmasked spot.
    """
    gen_length = int(kwargs.get("gen_length"))
    mask_id = int(kwargs.get("mask_id", 126336))
    
    gen_kwargs = dict(kwargs)
    gen_kwargs["steps"] = step_idx + 1
    
    # 1. Run Drafter
    drafter_out = drafter.generate(input_toks={"input_ids": current_tokens}, **gen_kwargs)
    
    # Simulate Drafter unmasking tokens
    drafted_tokens = sample_verifier(
        logits=drafter_out["logits"], 
        tokens=current_tokens, 
        mask_id=mask_id, 
        gen_length=gen_length, 
        steps=total_steps
    )
    
    # 2. Run Verifier on the Drafter's output
    ver_out = verifier.generate(input_toks={"input_ids": drafted_tokens}, **gen_kwargs)
    ver_logits = ver_out["logits"]
    
    # Get the verifier's predicted tokens (argmax)
    ver_predicted_ids = ver_logits.argmax(dim=-1)
    
    # Positions drafter unmasked
    drafted_positions = (current_tokens == mask_id) & (drafted_tokens != mask_id)
    
    # Check where Verifier's argmax MATCHES Drafter's choice
    agreed_positions = drafted_positions & (ver_predicted_ids == drafted_tokens)
    
    failed_count = drafted_positions.sum().item() - agreed_positions.sum().item()
    
    # Initialize next tokens
    next_tokens = current_tokens.clone()
    
    # Accept the ones they agreed on
    next_tokens[agreed_positions] = drafted_tokens[agreed_positions]
    
    # 4. Fallback: For the tokens where they disagreed (or Verifier wants its own top choices)
    if failed_count > 0:
        remaining_mask = next_tokens == mask_id
        ver_confidence = ver_logits.max(dim=-1).values
        ver_confidence = ver_confidence.masked_fill(~remaining_mask, float('-inf'))
        
        _, topk_indices = ver_confidence.topk(failed_count, dim=-1)
        next_tokens.scatter_(1, topk_indices, ver_predicted_ids.gather(1, topk_indices))
        
    return {
        "next_tokens": next_tokens,
        "metrics": {
            "drafter_nfe": 1,
            "verifier_nfe": 1,
            "drafted_count": drafted_positions.sum().item(),
            "accepted_count": agreed_positions.sum().item(),
            "acceptance_rate": agreed_positions.sum().item() / max(1, drafted_positions.sum().item())
        }
    }
