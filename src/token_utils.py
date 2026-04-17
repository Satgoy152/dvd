import torch

def convert_tokens(sequence_ids, src_tokenizer, dst_tokenizer, src_mask_id, dst_mask_id):
    """
    Translates a sequence of token IDs from a source tokenizer into a sequence of token IDs 
    for a destination tokenizer, explicitly preserving the exact counts and positions of MASK tokens.
    
    Args:
        sequence_ids: List[int] or torch.Tensor of source token ids
        src_tokenizer: HuggingFace Tokenizer (Drafter)
        dst_tokenizer: HuggingFace Tokenizer (Verifier or vice-versa)
        src_mask_id: int, the model-specific token ID for a MASK in the source
        dst_mask_id: int, the model-specific token ID for a MASK in the destination
        
    Returns:
        Converted tokens matching the type of `sequence_ids` (List[int] or torch.Tensor)
    """
    if isinstance(sequence_ids, torch.Tensor):
        original_shape = sequence_ids.shape
        seq = sequence_ids.flatten().tolist()
    else:
        original_shape = None
        seq = sequence_ids
        
    converted_seq = []
    current_chunk = []
    
    for token in seq:
        if token == src_mask_id:
            # We hit a mask. First, if we have a current_chunk, process it.
            if len(current_chunk) > 0:
                text_chunk = src_tokenizer.decode(current_chunk, skip_special_tokens=True)
                # Ensure we don't accidentally drop tokens if the decode returns empty for some reason, 
                # although typically it will return a string.
                if text_chunk: 
                   encoded_chunk = dst_tokenizer(text_chunk, add_special_tokens=False)["input_ids"]
                   converted_seq.extend(encoded_chunk)
                current_chunk = []
            
            # Now add the destination mask
            converted_seq.append(dst_mask_id)
        else:
            current_chunk.append(token)
            
    # Process any remaining non-mask tokens at the end
    if len(current_chunk) > 0:
        text_chunk = src_tokenizer.decode(current_chunk, skip_special_tokens=True)
        if text_chunk:
            encoded_chunk = dst_tokenizer(text_chunk, add_special_tokens=False)["input_ids"]
            converted_seq.extend(encoded_chunk)
            
    if original_shape and isinstance(sequence_ids, torch.Tensor):
        device = sequence_ids.device
        dtype = sequence_ids.dtype
        tensor_output = torch.tensor(converted_seq, device=device, dtype=dtype)
        # Preserve batch dimension if it was present
        if len(original_shape) == 2 and original_shape[0] == 1:
            tensor_output = tensor_output.unsqueeze(0) 
        return tensor_output
        
    return converted_seq
