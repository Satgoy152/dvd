def load_everything():
    return {"status": "ready"}

def custom_tokenize(prompt, state, tokenizer_max_len=50):
    # Return what we received so the test can check it
    return {"received_max_len": tokenizer_max_len}

def run_inference(tensors, state, drafter_steps=10, use_fp16=False, drafted_latents=None):
    # Return what we received
    return {
        "received_steps": drafter_steps,
        "received_fp16": use_fp16,
        "is_verifier_mode": drafted_latents is not None
    }