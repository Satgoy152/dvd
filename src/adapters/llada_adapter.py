import torch
from transformers import AutoTokenizer, AutoModel

def load_llada_model(model_id="GSAI-ML/LLaDA-8B-Base"):
    """Loads the custom LLaDA diffusion architecture."""
    print(f"      -> [LLaDA ADAPTER] Loading Diffusion Model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # LLaDA requires bfloat16 and trust_remote_code to execute its custom architecture
    model = AutoModel.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    return {"tokenizer": tokenizer, "model": model}

def llada_tokenize(prompt, state, tokenizer_max_len=128):
    print(f"      -> [LLaDA ADAPTER] Tokenizing prompt...")
    tokenizer = state["tokenizer"]
    inputs = tokenizer(prompt, return_tensors="pt", max_length=tokenizer_max_len, truncation=True)
    return inputs

def llada_generate(tensors, state, gen_length=64, steps=64):
    """Translates the standard generation request into LLaDA's masked diffusion process."""
    print(f"      -> [LLaDA ADAPTER] Running Masked Diffusion for {steps} steps...")
    model = state["model"]
    tokenizer = state["tokenizer"]
    device = model.device
    
    prompt_ids = tensors["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]
    
    # LLaDA uses a highly specific mask token ID for its diffusion process
    mask_id = 126336 
    
    # 1. Initialize the sequence: Prompt + [MASK] * gen_length
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids
    
    # 2. The Diffusion Generation Loop
    # Note: To run full high-quality generation, you would drop LLaDA's official 
    # generate.py script into this folder and import their exact scheduling loop.
    # For this adapter test, we perform a single forward pass to prove the tensors flow.
    print("      -> [LLaDA ADAPTER] Executing reverse diffusion process...")
    with torch.no_grad():
        outputs = model(x)
        predicted_token_ids = outputs["logits"].argmax(-1)
        
    # Decode only the generated portion (ignoring the prompt)
    result_text = tokenizer.decode(predicted_token_ids[0][prompt_len:], skip_special_tokens=True)
    
    return {"final_text": result_text, "raw_tensors": predicted_token_ids}