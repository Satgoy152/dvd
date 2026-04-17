import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_mdlm_model(model_id="dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"):
    print(f"      -> [MDLM ADAPTER] Loading Diffusion Model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    model = AutoModelForMaskedLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    
    # MDLM models based on Qwen often hide the mask token ID
    # Verify it here. If tokenizer.mask_token_id is None, we find it manually.
    if tokenizer.mask_token_id is None:
        # For Qwen-based MDLM, it's often the last token or a specific reserved token
        tokenizer.mask_token = "<|mask|>" 
        # If the above fails, check the config for 'mask_token_id'
        
    # ARCHITECTURE PATCH: 
    # This specific model requires the layers to be set to full_attention
    layers = getattr(model.model, "layers", []) if hasattr(model, "model") else getattr(model, "layers", [])
    for layer in layers:
        setattr(layer, "attention_type", "full_attention")

    if torch.cuda.is_available():
        model = model.cuda()
        
    model.eval()
    return {"tokenizer": tokenizer, "model": model}

def mdlm_tokenize(prompt, state, tokenizer_max_len=256, **kwargs):
    tokenizer = state["tokenizer"]
    # Instruct models MUST use the chat template to avoid 'hallucinating' Arabic characters
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(formatted, return_tensors="pt", max_length=tokenizer_max_len, truncation=True)

@torch.no_grad()
def mdlm_generate(tensors, state, gen_length=256, steps=0, **kwargs):
    model = state["model"]
    tokenizer = state["tokenizer"]
    device = model.device
    
    # Ensure input_ids are pulled correctly from the dict
    input_ids = tensors["input_ids"] if isinstance(tensors, dict) else tensors
    input_ids = input_ids.to(device)
    
    prompt_len = input_ids.shape[1]
    mask_id = tokenizer.mask_token_id

    # 1. Canvas Setup
    if steps == 0:
        # Initialize sequence: Prompt + [MASK] * gen_length
        x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids
    else:
        # Continue with existing sequence
        x = input_ids

    # 2. FORCE BIDIRECTIONAL ATTENTION
    # We create a 4D mask [batch, 1, seq_len, seq_len] of zeros (no masking)
    # or a 2D mask of ones. Qwen2 architecture usually expects 2D ones for 'full'
    attention_mask = torch.ones_like(x, device=device)
    
    # 3. Explicitly pass position_ids to ensure they don't reset in diffusion
    position_ids = torch.arange(x.shape[1], device=device).unsqueeze(0)

    outputs = model(
        input_ids=x, 
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    
    logits = outputs.logits.to(torch.float32)
    
    return {
        "logits": logits, 
        "tokens": logits.argmax(-1)
    }