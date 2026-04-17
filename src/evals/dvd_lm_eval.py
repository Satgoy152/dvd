from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
import torch

# Update sys.path to allow absolute imports from src
import sys
import os

from src.registry import ModelRegistry
from src.model import ModelBroker
from src.runtime import run

class DVD_LM(LM):
    def __init__(
        self,
        drafter_name: str,
        verifier_name: str,
        algorithm_name: str,
        steps: int = 10,
        gen_length: int = 64,
        batch_size: int = 1,
        registry_path: str = "registry.yaml",
        **kwargs
    ):
        super().__init__()
        
        self.steps = steps
        self.gen_length = gen_length
        self.batch_size = batch_size
        self.custom_kwargs = {
            "steps": steps,
            "gen_length": gen_length,
            "batch_size": batch_size,
            **kwargs,
        }
        
        print(f"[DVD EVAL] Initializing DVD LM Wrapper (batch_size={batch_size})...")
        self.registry = ModelRegistry(registry_path)
        
        d_mod, d_conf = self.registry.load_model(drafter_name)
        self.drafter = ModelBroker(d_mod, d_conf, role="DRAFTER")
        
        v_mod, v_conf = self.registry.load_model(verifier_name)
        self.verifier = ModelBroker(v_mod, v_conf, role="VERIFIER")
        
        self.algorithm_func = self.registry.load_algorithm(algorithm_name)
        
        self.total_aggregated_metrics = {}

    def generate_until(self, requests) -> list[str]:
        """
        Executes generation for a batch of requests (each request has `args=(context, until)`).
        Uses the batched runtime to process all requests together.
        """
        # Collect all prompts from the request batch
        batch_prompts = [request.args[0] for request in requests]

        # Run batched DVD inference
        output = run(
            drafter=self.drafter,
            verifier=self.verifier,
            algorithm_func=self.algorithm_func,
            prompts=batch_prompts,
            custom_kwargs=self.custom_kwargs,
        )

        # Track aggregate metrics across all generate_until calls
        for k, v in output["aggregated_metrics"].items():
            self.total_aggregated_metrics[k] = self.total_aggregated_metrics.get(k, 0) + v
        self.total_aggregated_metrics["request_count"] = (
            self.total_aggregated_metrics.get("request_count", 0) + len(batch_prompts)
        )

        texts = [r["text"] for r in output["results"]]
        print(f"[DVD EVAL] Batch of {len(texts)} complete. Progress Metrics: {self.total_aggregated_metrics}")
        return texts

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """
        DVD uses masked diffusion, which doesn't provide standard AR loglikelihoods easily.
        If evaluations strictly require loglikelihood, this raises NotImplementedError.
        """
        raise NotImplementedError("DVD does not currently support standard AR loglikelihood computation.")
        
    def loglikelihood_rolling(self, requests) -> list[float]:
        raise NotImplementedError("DVD does not support rolling loglikelihood.")
