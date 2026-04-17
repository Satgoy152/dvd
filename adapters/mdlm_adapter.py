import torch
import transformers.processing_utils
import transformers.utils
import transformers.models.auto.auto_factory as auto_factory
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ====================================================================
# THE "GHOST IMPORT" PATCHES (For Locked Transformers Environments)
# ====================================================================
# 1. Fake the 'Unpack' typing feature
if getattr(transformers.processing_utils, "Unpack", None) is None:
    class DummyUnpack:
        def __getitem__(self, item): return item
    transformers.processing_utils.Unpack = DummyUnpack()

# 2. Fake the TransformersKwargs dictionary
if getattr(transformers.utils, "TransformersKwargs", None) is None:
    transformers.utils.TransformersKwargs = dict

# 3. Fake the Flex Attention checker (Forces standard attention)
if getattr(transformers.utils, "is_torch_flex_attn_available", None) is None:
    transformers.utils.is_torch_flex_attn_available = lambda: False

# 4. Bypass the strict config_class Registration Error
if not hasattr(auto_factory._BaseAutoModelClass, "_dvd_register_patched"):
    original_register = auto_factory._BaseAutoModelClass.register

    @classmethod
    def patched_register(cls, config_class, model_class, exist_ok=False):
        # Temporarily overwrite the internal name so the strict check passes
        saved_config = getattr(model_class, "config_class", None)
        model_class.config_class = config_class
        try:
            return original_register.__func__(cls, config_class, model_class, exist_ok=exist_ok)
        finally:
            # Put it back exactly how we found it
            if saved_config is not None:
                model_class.config_class = saved_config

    auto_factory._BaseAutoModelClass.register = patched_register
    auto_factory._BaseAutoModelClass._dvd_register_patched = True
# ====================================================================


def load_mdlm_model(model_id="dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"):
    print(f"      -> [MDLM ADAPTER] Loading Diffusion Model '{model_id}'...")
    
    # Bypass the Rust tokenization error by using the identical Qwen2 vocab
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",  
        trust_remote_code=True
    )
    
    model = AutoModelForMaskedLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False
    )
    
    if getattr(tokenizer, "mask_token_id", None) is None:
        tokenizer.add_special_tokens({'mask_token': '<|mask|>'}) 
        
    # ==========================================================
    # ARCHITECTURE PATCH for MDLM
    # ==========================================================
    base_model = getattr(model, "model", model)
    
    # 1. Inject the missing sliding layers flag so the forward pass doesn't crash
    if not hasattr(base_model, "has_sliding_layers"):
        base_model.has_sliding_layers = False
        
    # 2. Force full attention (Your existing patch)
    layers = getattr(base_model, "layers", [])
    for layer in layers:
        setattr(layer, "attention_type", "full_attention")
    # ==========================================================

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
def mdlm_generate(tensors, state, **kwargs):
    """
    Pure Forward Pass for MDLM. 
    The canvas (Prompt + Masks) is entirely managed by the DVD framework.
    """
    model = state["model"]
    device = model.device
    
    # 1. Safely extract inputs provided by `_common.py`
    if isinstance(tensors, dict):
        input_ids = tensors["input_ids"].to(device)
        # We can use the attention mask provided by the framework
        attention_mask = tensors.get("attention_mask", torch.ones_like(input_ids)).to(device)
    else:
        input_ids = tensors.to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

    # 2. Explicitly pass position_ids to ensure they don't reset in diffusion
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    # 3. Standard Forward Pass
    outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    
    logits = outputs.logits.to(torch.float32)
    
    return {
        "logits": logits, 
        "tokens": logits.argmax(-1)
    }