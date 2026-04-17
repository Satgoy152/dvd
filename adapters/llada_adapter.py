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
    initialize=False,
    gen_length=None,
    mask_id=LLADA_MASK_ID,
    prompt_len=None,
    **kwargs,
):
    """Run one forward pass; return logits over the generation region only.

    Args:
        tensors: dict with "input_ids" (B, L) content-only ids (no template),
                 and optional "attention_mask" (B, L) marking pad positions.
                 - initialize=True:  L == max_prompt_len (prompt only)
                 - initialize=False: L == max_prompt_len + gen_length
        initialize: if True, append gen_length mask tokens to the prompt.
        gen_length: number of tokens in the generation region.
        mask_id:    mask token id used to fill the new gen region.
        prompt_len: length of the prompt portion when initialize=False.

    Returns:
        {"logits": (B, gen_length, V), "tokens": None}
    """
    model = state["model"]
    device = model.device

    content = tensors["input_ids"].to(device)
    content_attn = tensors.get("attention_mask")
    if content_attn is not None:
        content_attn = content_attn.to(device)

    if initialize:
        if gen_length is None:
            raise ValueError("llada_generate: gen_length required when initialize=True")
        prompt_ids = content
        B = content.shape[0]
        gen_region = torch.full((B, gen_length), mask_id, dtype=torch.long, device=device)
        prompt_attn = content_attn if content_attn is not None else None
    else:
        if prompt_len is None:
            raise ValueError("llada_generate: prompt_len required when initialize=False")
        if gen_length is None:
            gen_length = content.shape[1] - prompt_len
        prompt_ids = content[:, :prompt_len]
        gen_region = content[:, prompt_len:]
        prompt_attn = content_attn[:, :prompt_len] if content_attn is not None else None

    prefix_ids = state["template_prefix_ids"].to(device)
    suffix_ids = state["template_suffix_ids"].to(device)
    B = prompt_ids.shape[0]
    prefix_batch = prefix_ids.unsqueeze(0).expand(B, -1)
    suffix_batch = suffix_ids.unsqueeze(0).expand(B, -1)

    x = torch.cat([prefix_batch, prompt_ids, suffix_batch, gen_region], dim=1)

    if prompt_attn is not None:
        ones = lambda n: torch.ones((B, n), dtype=prompt_attn.dtype, device=device)
        attn_mask = torch.cat([ones(prefix_ids.shape[0]), prompt_attn,
                               ones(suffix_ids.shape[0]), ones(gen_length)], dim=1)
    else:
        attn_mask = None

    with torch.no_grad():
        outputs = model(x, attention_mask=attn_mask) if attn_mask is not None else model(x)

    logits = outputs["logits"]
    gen_logits = logits[:, -gen_length:, :]
    return {"logits": gen_logits, "tokens": None}
