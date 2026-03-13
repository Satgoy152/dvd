import sys
import os
import yaml

# This allows our test to import model.py and registry.py from the folder above
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from registry import ModelRegistry
from model import ModelBroker

def test_smart_kwargs_filtering(tmp_path):
    """
    Tests that the ModelBroker correctly filters **kwargs using inspect, 
    so functions only receive the arguments they explicitly ask for.
    """
    # ---------------------------------------------------------
    # 1. ARRANGE: Create a temporary registry.yaml for this test
    # (tmp_path is a magic Pytest feature that makes a temp folder)
    # ---------------------------------------------------------
    registry_file = tmp_path / "test_registry.yaml"
    test_config = {
        "available_models": [{
            "name": "test_model",
            "path": "./tests/dummy_model.py",
            "init_func": "load_everything",
            "tokenizer_func": "custom_tokenize",
            "gen_func": "run_inference"
        }]
    }
    
    with open(registry_file, 'w') as f:
        yaml.dump(test_config, f)

    # Load the registry and broker
    registry = ModelRegistry(yaml_path=str(registry_file))
    module, config = registry.load_model("test_model")
    broker = ModelBroker(module, config, role="TESTER")

    cli_kwargs = {
        "tokenizer_max_len": 99,
        "drafter_steps": 25,
        "use_fp16": True,
        "random_unused_arg": "this should be ignored"
    }

    # ---------------------------------------------------------
    # 2. ACT: Run the broker methods
    # ---------------------------------------------------------
    tensors = broker.tokenize("Hello World", **cli_kwargs)
    latents = broker.generate(tensors, **cli_kwargs)

    # ---------------------------------------------------------
    # 3. ASSERT: Verify the smart filtering worked
    # ---------------------------------------------------------
    
    # Check Tokenizer Output
    # It should have updated the default 50 to 99
    assert tensors["received_max_len"] == 99
    
    # Check Generator Output
    # It should have received 25 and True, and ignored the unused arg without crashing
    assert latents["received_steps"] == 25
    assert latents["received_fp16"] is True
    assert latents["is_verifier_mode"] is False # Because we didn't pass drafted_latents