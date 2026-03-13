import inspect

class ModelBroker:
    def __init__(self, module, config, role="Model"):
        self.config = config
        self.role = role
        
        self.init_func = getattr(module, self.config['init_func'])
        self.tokenizer_func = getattr(module, self.config['tokenizer_func'])
        self.gen_func = getattr(module, self.config['gen_func'])
        
        print(f"[{self.role}] Initializing weights and tokenizer...")
        
        # --- NEW LOGIC HERE ---
        # Route the YAML config through the smart filter so the init_func 
        # can receive variables (like model_id) if it asks for them.
        init_kwargs = self._filter_kwargs(self.init_func, self.config)
        self.model_state = self.init_func(**init_kwargs)

    def _filter_kwargs(self, target_func, all_kwargs):
        """Filters the giant kwargs pool to only include what the function expects."""
        sig = inspect.signature(target_func)
        
        # If the user's function specifically accepts **kwargs, give them everything
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_varkw:
            return all_kwargs
            
        # Otherwise, strictly filter the dictionary based on their parameter names
        valid_keys = [p.name for p in sig.parameters.values()]
        filtered_kwargs = {k: v for k, v in all_kwargs.items() if k in valid_keys}
        
        return filtered_kwargs

    def tokenize(self, prompt: str, **kwargs):
        print(f"[{self.role}] Tokenizing prompt...")
        # Only pass kwargs that the tokenizer function actually asked for
        tokenizer_kwargs = self._filter_kwargs(self.tokenizer_func, kwargs)
        return self.tokenizer_func(prompt, self.model_state, **tokenizer_kwargs)

    def generate(self, input_toks, drafted_latents=None, **kwargs):
        print(f"[{self.role}] Running inference...")
        # Only pass kwargs that the generation function actually asked for
        gen_kwargs = self._filter_kwargs(self.gen_func, kwargs)
        
        # We handle drafted_latents explicitly. If they exist (Verifier), pass them in.
        return self.gen_func(input_toks, self.model_state, **gen_kwargs)