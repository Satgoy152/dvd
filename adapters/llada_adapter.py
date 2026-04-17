import torch
from transformers import AutoTokenizer, AutoModel

LLADA_MASK_ID = 126336
PLACEHOLDER = "@@DVD_PROMPT_PLACEHOLDER@@"


def load_llada_model(model_id="GSAI-ML/LLaDA-8B-Instruct"):
    print(f"      -> [LLaDA ADAPTER] Loading Diffusion Model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    if torch.cuda.is_available():
        model = model.cuda()

    prefix_ids, suffix_ids = _compute_template_ids(tokenizer)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "template_prefix_ids": prefix_ids,
        "template_suffix_ids": suffix_ids,
    }


def _compute_template_ids(tokenizer):
    # Cache chat-template prefix/suffix in token-id space so generate can wrap
    # content cheaply each step (no text decode/encode roundtrip).
    if getattr(tokenizer, "chat_template", None) is None:
        bos = getattr(tokenizer, "bos_token_id", None)
        prefix = torch.tensor([bos], dtype=torch.long) if bos is not None else torch.empty(0, dtype=torch.long)
        return prefix, torch.empty(0, dtype=torch.long)

    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": PLACEHOLDER}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if PLACEHOLDER not in templated:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    prefix_text, suffix_text = templated.split(PLACEHOLDER, 1)
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
    return prefix_ids, suffix_ids


def llada_tokenize(prompt, state, tokenizer_max_len=128, **kwargs):
    """Pure text -> token ids. No chat template, no mask padding.

    Returns {"input_ids": (1, L)}.
    """
    tokenizer = state["tokenizer"]
    return tokenizer(
        prompt,
        return_tensors="pt",
        max_length=tokenizer_max_len,
        truncation=True,
        add_special_tokens=False,
    )


def llada_generate(
    tensors,
    state,
    gen_length=256,
    mask_id=LLADA_MASK_ID,
    **kwargs,
):
    """
    Framework-compliant forward pass.
    Handles chat templates natively and pads logits to match _common.py's expectations.
    """
    model = state["model"]
    device = model.device

    content = tensors["input_ids"].to(device)
    content_attn = tensors.get("attention_mask")
    if content_attn is not None:
        content_attn = content_attn.to(device)

    # 1. Dynamically calculate prompt length based on the framework's canvas
    prompt_len = kwargs.get("prompt_len")
    if prompt_len is None:
        prompt_len = content.shape[1] - gen_length

    # Split the framework's canvas
    prompt_ids = content[:, :prompt_len]
    gen_region = content[:, prompt_len:]

    # 2. Safely Inject the Chat Template (Prefix and Suffix)
    prefix_ids = state["template_prefix_ids"].to(device)
    suffix_ids = state["template_suffix_ids"].to(device)
    B = prompt_ids.shape[0]
    prefix_batch = prefix_ids.unsqueeze(0).expand(B, -1)
    suffix_batch = suffix_ids.unsqueeze(0).expand(B, -1)

    # Reconstruct the sequence for the model
    x = torch.cat([prefix_batch, prompt_ids, suffix_batch, gen_region], dim=1)

    if content_attn is not None:
        prompt_attn = content_attn[:, :prompt_len]
        ones = lambda n: torch.ones((B, n), dtype=prompt_attn.dtype, device=device)
        attn_mask = torch.cat([ones(prefix_ids.shape[0]), prompt_attn,
                               ones(suffix_ids.shape[0]), ones(gen_length)], dim=1)
    else:
        attn_mask = None

    # 3. Standard Forward Pass
    with torch.no_grad():
        outputs = model(x, attention_mask=attn_mask) if attn_mask is not None else model(x)

    # 4. Extract only the generated logits
    logits = outputs["logits"]
    gen_logits = logits[:, -gen_length:, :]
    
    # 5. CRITICAL FIX: Pad the left side with dummy logits!
    # This ensures the tensor shape perfectly matches the `prompt_len + gen_length` 
    # canvas size so _common.py can slice it safely without crashing.
    dummy = torch.zeros((B, prompt_len, gen_logits.shape[-1]), dtype=gen_logits.dtype, device=device)
    full_framework_logits = torch.cat([dummy, gen_logits], dim=1)

    return {"logits": full_framework_logits, "tokens": None}