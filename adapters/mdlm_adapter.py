import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_mdlm_model(model_id="dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"):
    print(f"      -> [MDLM ADAPTER] Loading Diffusion Model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        
    model = AutoModelForMaskedLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        dtype=torch.bfloat16,
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
        
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return {"tokenizer": tokenizer, "model": model}

def mdlm_tokenize(prompt, state, tokenizer_max_len=256):
    print(f"      -> [MDLM ADAPTER] Tokenizing prompt with Chat Template...")
    tokenizer = state["tokenizer"]
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", max_length=tokenizer_max_len, truncation=True)
    return inputs

@torch.no_grad()
def mdlm_generate(tensors, state, **kwargs):
    model = state["model"]
    tokenizer = state["tokenizer"]
    device = model.device 
    
    # 1. Grab the sequence provided by baseline.py
    x = tensors["input_ids"].to(device)

    # 2. DYNAMIC MASK SWAPPING (The Fix for the Gibberish)
    # Convert the framework's empty slots (<|endoftext|>) into real MDLM masks.
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    
    if pad_id is not None and mask_id is not None and pad_id != mask_id:
        x = torch.where(x == pad_id, mask_id, x)

    # 3. ARCHITECTURE PATCH (The Fix for the AttributeError)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if not hasattr(layer, "attention_type"):
                setattr(layer, "attention_type", "full_attention")

    # 4. PURE FORWARD PASS
    # Crucially, we do NOT pass attention_mask or use_cache here. 
    # We let the custom modeling_qwen2.py handle it internally just like the HF script.
    outputs = model(x)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    
    # 5. Prevent NaN collapse
    logits = logits.to(torch.float32)
    predicted_token_ids = logits.argmax(-1)
    
    return {
        "logits": logits, 
        "tokens": predicted_token_ids
    }