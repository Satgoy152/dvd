import torch


def tokenize_batch(broker, prompts, **kwargs):
    """Tokenize each prompt via `broker` and right-pad to uniform length.

    Returns:
        content:     (B, max_prompt_len) token ids, right-padded with pad_id
        content_attn:(B, max_prompt_len) 1 for real, 0 for pad
        prompt_len:  int, max_prompt_len across the batch
    """
    tokenizer = broker.model_state["tokenizer"]
    device = broker.model_state["model"].device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    seqs, lengths = [], []
    for p in prompts:
        toks = broker.tokenize(p, **kwargs)["input_ids"].squeeze(0)
        seqs.append(toks)
        lengths.append(toks.shape[0])

    max_len = max(lengths)
    B = len(prompts)
    content = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, (seq, plen) in enumerate(zip(seqs, lengths)):
        content[i, :plen] = seq.to(device)
        attn[i, :plen] = 1
    return content, attn, max_len


def initialize_content(broker, prompts, gen_length, mask_id, **kwargs):
    """Tokenize prompts and run broker.generate(initialize=True).

    Returns:
        full_content: (B, prompt_len + gen_length) with gen region all mask_id
        full_attn:    (B, prompt_len + gen_length)
        prompt_len:   int
        logits:       (B, gen_length, V) from the initial forward pass
    """
    prompt_ids, prompt_attn, prompt_len = tokenize_batch(broker, prompts, **kwargs)
    out = broker.generate(
        input_toks={"input_ids": prompt_ids, "attention_mask": prompt_attn},
        initialize=True,
        gen_length=gen_length,
        mask_id=mask_id,
    )
    B = prompt_ids.shape[0]
    device = prompt_ids.device
    gen_region = torch.full((B, gen_length), mask_id, dtype=torch.long, device=device)
    full_content = torch.cat([prompt_ids, gen_region], dim=1)
    full_attn = torch.cat([prompt_attn, torch.ones_like(gen_region)], dim=1)
    return full_content, full_attn, prompt_len, out["logits"]


def step_generate(broker, full_content, full_attn, prompt_len, gen_length, mask_id):
    """Run broker.generate(initialize=False) on an already-initialized state.

    Returns logits (B, gen_length, V).
    """
    out = broker.generate(
        input_toks={"input_ids": full_content, "attention_mask": full_attn},
        initialize=False,
        gen_length=gen_length,
        mask_id=mask_id,
        prompt_len=prompt_len,
    )
    return out["logits"]


def sample_topk_confident(logits, gen_region, mask_id, k):
    """Unmask the top-k most confident still-masked positions per batch item.

    Returns an updated gen_region tensor.
    """
    predicted_ids = logits.argmax(dim=-1)
    confidence = logits.max(dim=-1).values
    mask_positions = gen_region == mask_id
    confidence = confidence.masked_fill(~mask_positions, float("-inf"))
    _, topk_indices = confidence.topk(k, dim=-1)
    updated = gen_region.clone()
    updated.scatter_(1, topk_indices, predicted_ids.gather(1, topk_indices))
    return updated


def decode_results(broker, full_content, prompts, prompt_len):
    """Decode the generation region of each batch item to text."""
    tokenizer = broker.model_state["tokenizer"]
    results = []
    for i in range(full_content.shape[0]):
        gen_tokens = full_content[i, prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        results.append({
            "text": text,
            "tokens": full_content[i:i + 1].clone(),
            "prompt": prompts[i],
        })
    return results
