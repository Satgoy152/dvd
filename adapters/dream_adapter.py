import torch
import transformers
from transformers import AutoTokenizer, AutoModel

# ====================================================================
# Modern Hugging Face automatically injects security kwargs.
# Older models pass these blindly to their __init__ and crash.
# ====================================================================
if not hasattr(transformers.PreTrainedModel, "_dvd_dream_patched"):
    original_from_pretrained = transformers.PreTrainedModel.from_pretrained

    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Quietly delete the offending modern keyword argument
        kwargs.pop("weights_only", None)
        
        # Continue the normal Hugging Face loading process
        return original_from_pretrained.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    transformers.PreTrainedModel.from_pretrained = patched_from_pretrained
    transformers.PreTrainedModel._dvd_dream_patched = True
# ====================================================================

def load_dream_model(model_id="Dream-org/Dream-v0-Instruct-7B"):
    print(f"      -> [DREAM ADAPTER] Loading Native MDLM '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if getattr(tokenizer, "mask_token_id", None) is None:
        tokenizer.add_special_tokens({'mask_token': '<|mask|>'}) 
        
    model = AutoModel.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    model.eval()
    return {"tokenizer": tokenizer, "model": model}

def dream_tokenize(prompt, state, tokenizer_max_len=256, **kwargs):
    tokenizer = state["tokenizer"]
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        return_dict=True, 
        add_generation_prompt=True
    )
    return inputs

@torch.no_grad()
def dream_generate(tensors, state, **kwargs):
    """A pure, bare-metal forward pass. No internal loops."""
    model = state["model"]
    device = model.device

    # Safely extract input_ids
    if isinstance(tensors, dict):
        input_ids = tensors['input_ids'].to(device)
    else:
        input_ids = tensors.to(device)

    # Standard Transformer forward pass
    outputs = model(input_ids=input_ids, return_dict=True)
    
    # Return real logits to your framework
    logits = outputs.logits.to(torch.float32)
    return {
        "logits": logits, 
        "tokens": logits.argmax(-1)
    }